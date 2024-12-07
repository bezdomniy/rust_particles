use glam::{Vec2, Vec3};
use itertools::{partition, Itertools};
use std::f32::{consts::PI, INFINITY, NEG_INFINITY};

use super::super::app::Particle;

static MAX_SHAPES_IN_NODE: usize = 4;

pub struct Bvh(pub Vec<NodeInner>);

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum SplitMethod {
    Middle,
    EqualCounts,
    Sah,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct NodeInner {
    pub centre: Vec2,
    pub radius: f32,
    pub skip_ptr_or_prim_idx1: u32,
    pub prim_idx2: u32,
}

#[derive(Debug, Default, Copy, Clone)]
struct AABB {
    first: Vec2,
    second: Vec2,
}

fn point_in_circle(centre: Vec2, radius: f32, point: Vec2) -> bool {
    centre.distance_squared(point) <= radius * radius
    // def in_circle(center_x, center_y, radius, x, y):
    // square_dist = (center_x - x) ** 2 + (center_y - y) ** 2
    // return square_dist <= radius ** 2
}

#[derive(Debug, Default, Copy, Clone)]
struct MortonPrimitive {
    primitive_index: usize,
    morton_code: u32,
}

fn radix_sort(inp: &mut Vec<MortonPrimitive>) -> Vec<MortonPrimitive> {
    let mut out = inp.clone();
    const BITS_PER_PASS: u32 = 6;
    const N_BITS: u32 = 30;
    const N_PASSES: u32 = N_BITS / BITS_PER_PASS;
    const BIT_MASK: u32 = (1 << BITS_PER_PASS) - 1;
    const N_BUCKETS: usize = 1 << BITS_PER_PASS;

    (0..N_PASSES).for_each(|pass| {
        if pass > 0 {
            std::mem::swap(inp, &mut out);
        };

        let low_bit: u32 = pass * BITS_PER_PASS;
        let mut bucket_count = [0; N_BUCKETS];

        inp.iter().for_each(|morton_primitive| {
            let bucket = ((morton_primitive.morton_code >> low_bit) & BIT_MASK) as usize;
            bucket_count[bucket] += 1;
        });

        let mut out_index = [0; N_BUCKETS];
        for i in 1..out_index.len() {
            out_index[i] = out_index[i - 1] + bucket_count[i - 1];
        }

        inp.iter().for_each(|morton_primitive| {
            let bucket = ((morton_primitive.morton_code >> low_bit) & BIT_MASK) as usize;
            out[out_index[bucket]] = *morton_primitive;
            out_index[bucket] += 1;
        });
    });

    out
}

fn left_shift_3(inp: u32) -> u32 {
    let mut x = inp;

    if x == (1 << 10) {
        x -= 1;
    }
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;
    x
}

fn encode_morton_3(inp: Vec3) -> u32 {
    (left_shift_3(inp.z as u32) << 2)
        | (left_shift_3(inp.y as u32) << 1)
        | left_shift_3(inp.x as u32)
}

impl NodeInner {
    pub fn new(centre: Vec2, radius: f32, skip_ptr_or_prim_idx1: u32, prim_idx2: u32) -> Self {
        NodeInner {
            centre: centre,
            radius: radius,
            skip_ptr_or_prim_idx1,
            prim_idx2,
        }
    }

    pub fn empty() -> Self {
        NodeInner {
            centre: Vec2::new(f32::INFINITY, f32::INFINITY),
            radius: 0f32,
            skip_ptr_or_prim_idx1: 0u32,
            prim_idx2: 0u32,
        }
    }

    pub fn merge(&self, other: &NodeInner) -> Self {
        if self.centre.x < INFINITY {
            let d = other.centre.distance(self.centre);

            if d + self.radius <= other.radius {
                return *other;
            }
            if d + other.radius <= self.radius {
                return *self;
            }

            let new_centre = (self.centre + other.centre) / 2f32;
            let new_radius = f32::max(self.radius, other.radius) + d / 2f32;

            // println!("b: {:?}", new_radius);

            return NodeInner::new(
                new_centre,
                new_radius,
                other.skip_ptr_or_prim_idx1,
                other.prim_idx2,
            );
        } else {
            return NodeInner::new(
                other.centre,
                other.radius,
                other.skip_ptr_or_prim_idx1,
                other.prim_idx2,
            );
        }
    }

    pub fn surface_area(&self) -> f32 {
        (self.radius * self.radius) * PI
    }
}

impl Bvh {
    pub fn empty() -> Self {
        Bvh(vec![])
    }

    pub fn new(particles: &mut [Particle], radius: f32, linear: bool) -> Self {
        let mut object_inner_nodes: Vec<NodeInner> =
            Vec::with_capacity(particles.len().next_power_of_two());

        if linear {
            Bvh::build_linear(&mut object_inner_nodes, particles, radius);
        } else {
            let split_method = SplitMethod::Sah;

            Bvh::recursive_build(
                &mut object_inner_nodes,
                particles,
                radius,
                0,
                particles.len(),
                split_method,
            );
        }

        if object_inner_nodes.is_empty() {
            return Bvh::empty();
        }
        Bvh(object_inner_nodes)
    }

    fn build_linear(
        bounding_circles: &mut Vec<NodeInner>,
        particles: &mut [Particle],
        radius: f32,
    ) -> u32 {
        let bounds = particles.iter().fold(AABB::empty(), |acc, new| {
            acc.add_point(&new.bounds_centroid())
        });

        const MORTON_BITS: u32 = 10;
        const MORTON_SCALE: f32 = (1 << MORTON_BITS) as f32;

        let mut morton_primitives = (0..particles.len())
            .map(|i| {
                let centroid_offset = bounds.offset(&particles[i].pos);
                let offset = centroid_offset * MORTON_SCALE;
                MortonPrimitive {
                    primitive_index: i,
                    morton_code: encode_morton_3(Vec3::new(offset.x, offset.y, 0f32)),
                }
            })
            .collect_vec();

        morton_primitives = radix_sort(&mut morton_primitives);

        const FIRST_BIT_INDEX: i32 = 29 - 12;

        let size = particles.len();

        // let particles: Vec<Particle> = particles.iter().map(|p| p.clone()).collect();
        let ordered_particles = morton_primitives
            .iter()
            .map(|mp| particles[mp.primitive_index])
            .collect_vec();

        particles.copy_from_slice(&ordered_particles);

        Bvh::emit_lbvh(
            morton_primitives.as_mut_slice(),
            particles,
            bounding_circles,
            radius,
            0,
            size,
            FIRST_BIT_INDEX,
        );

        bounding_circles.len() as u32
    }

    fn emit_lbvh(
        morton_primitives: &mut [MortonPrimitive],
        ordered_particles: &[Particle],
        bounding_circles: &mut Vec<NodeInner>,
        radius: f32,
        start: usize,
        end: usize,
        bit_index: i32,
    ) -> u32 {
        // println!("{:?}", bit_index);
        let n_primitives = end - start;
        let is_leaf: bool = bit_index == -1 || n_primitives <= 2;

        let mut bounds = ordered_particles[start..end]
            .iter()
            .fold(NodeInner::empty(), |acc: NodeInner, new| {
                acc.merge(&new.bounds(radius))
            });

        if is_leaf {
            // println!("leaf, {} {}", bit_index, n_primitives);
            bounds.skip_ptr_or_prim_idx1 = start as u32;
            bounds.prim_idx2 = end as u32;
            bounding_circles.push(bounds);
        } else {
            bounds.prim_idx2 = 0;

            let mask = 1 << bit_index;
            if (morton_primitives[start].morton_code & mask)
                == (morton_primitives[end - 1].morton_code & mask)
            {
                return Bvh::emit_lbvh(
                    morton_primitives,
                    ordered_particles,
                    bounding_circles,
                    radius,
                    start,
                    end,
                    bit_index - 1,
                );
            }

            let mut search_start = start;
            let mut search_end = end - 1;
            while search_start + 1 != search_end {
                let mid = (search_start + search_end) / 2;
                if (morton_primitives[search_start].morton_code & mask)
                    == (morton_primitives[mid].morton_code & mask)
                {
                    search_start = mid;
                } else {
                    search_end = mid;
                }
            }
            let split_offset = search_end;

            // let split_offset = (start + end) / 2;

            let curr_idx = bounding_circles.len();
            bounding_circles.push(bounds);

            Bvh::emit_lbvh(
                morton_primitives,
                ordered_particles,
                bounding_circles,
                radius,
                start,
                split_offset,
                bit_index - 1,
            );
            let skip_ptr = Bvh::emit_lbvh(
                morton_primitives,
                ordered_particles,
                bounding_circles,
                radius,
                split_offset,
                end,
                bit_index - 1,
            );

            bounding_circles[curr_idx].skip_ptr_or_prim_idx1 = skip_ptr;
        }
        bounding_circles.len() as u32
    }

    // TODO: make this return the skip pointer so it can bubble up
    fn recursive_build(
        bounding_circles: &mut Vec<NodeInner>,
        particles: &mut [Particle],
        radius: f32,
        start: usize,
        end: usize,
        split_method: SplitMethod,
    ) -> u32 {
        // println!("{} {}", start, end);
        // for x in &primitives[start..end] {
        //     println!("{}, {}, {:?}", start, end, x);
        // }
        // println!("start end: {:?} {:?}", start, end);

        let centroid_bounds = particles[start..end]
            .iter()
            .fold(AABB::empty(), |acc, new| {
                acc.add_point(&new.bounds_centroid())
            });

        let mut bounds = particles[start..end]
            .iter()
            .fold(NodeInner::empty(), |acc, new| {
                acc.merge(&new.bounds(radius))
            });

        // println!("b: {:?}", bounds);

        let diagonal = centroid_bounds.diagonal();

        let split_dimension = if diagonal.x > diagonal.y { 0 } else { 1 };

        let n_primitives = end - start;
        // let mid = (start + end) / 2;

        let is_leaf: bool = centroid_bounds.first[split_dimension]
            == centroid_bounds.second[split_dimension]
            || n_primitives <= 2;

        if is_leaf {
            // println!("leaf");
            bounds.skip_ptr_or_prim_idx1 = start as u32;
            bounds.prim_idx2 = end as u32;
            bounding_circles.push(bounds);
        } else {
            // bounds.skip_ptr_or_prim_idx1 = 2u32.pow((bvh_height - level) as u32) - 1;
            bounds.prim_idx2 = 0;

            let mut fallthrough = false;
            let mut mid = (start + end) / 2;

            if matches!(split_method, SplitMethod::Middle) {
                let pmid = (centroid_bounds.first[split_dimension]
                    + centroid_bounds.second[split_dimension])
                    / 2f32;

                mid = partition(particles[start..end].iter_mut(), |n| {
                    n.bounds_centroid()[split_dimension] < pmid
                }) + start;

                if mid != start && mid != end {
                    fallthrough = true;
                }
            }

            if fallthrough || matches!(split_method, SplitMethod::EqualCounts) {
                mid = (start + end) / 2;
                particles[start..end].select_nth_unstable_by(mid - start, |a, b| {
                    a.bounds_centroid()[split_dimension]
                        .partial_cmp(&b.bounds_centroid()[split_dimension])
                        .unwrap()
                });
            }

            if matches!(split_method, SplitMethod::Sah) {
                if n_primitives <= 2 {
                    mid = (start + end) / 2;
                    particles[start..end].select_nth_unstable_by(mid - start, |a, b| {
                        a.bounds_centroid()[split_dimension]
                            .partial_cmp(&b.bounds_centroid()[split_dimension])
                            .unwrap()
                    });
                } else {
                    let n_buckets: usize = 12;
                    let mut buckets = vec![NodeInner::empty(); n_buckets];

                    for triangle in particles.iter().take(end).skip(start) {
                        let mut b: usize = n_buckets
                            * centroid_bounds.offset(&triangle.bounds_centroid())[split_dimension]
                                .round() as usize;

                        if b == n_buckets {
                            b = n_buckets - 1;
                        };

                        buckets[b].skip_ptr_or_prim_idx1 += 1; // Using this for the count variable
                        buckets[b].merge(&triangle.bounds(radius));
                    }

                    let mut cost = vec![0f32; n_buckets - 1];

                    for (i, c) in cost.iter_mut().enumerate().take(n_buckets - 1) {
                        let mut b0 = NodeInner::empty();
                        let mut b1 = NodeInner::empty();
                        let mut count0: u32 = 0;
                        let mut count1: u32 = 0;

                        for node in buckets.iter().take(i + 1) {
                            b0 = b1.merge(node);
                            count0 += node.skip_ptr_or_prim_idx1;
                        }

                        for node in buckets.iter().take(n_buckets).skip(i + 1) {
                            b1 = b1.merge(node);
                            count1 += node.skip_ptr_or_prim_idx1;
                        }

                        *c = 1f32
                            + (count0 as f32 * b0.surface_area()
                                + count1 as f32 * b1.surface_area())
                                / bounds.surface_area();
                    }

                    let mut min_cost = cost[0];
                    let mut min_cost_split_bucket: usize = 0;

                    for (i, c) in cost.iter().enumerate().take(n_buckets - 1).skip(1) {
                        if *c < min_cost {
                            min_cost = *c;
                            min_cost_split_bucket = i;
                        }
                    }

                    let leaf_cost = n_primitives as f32;
                    if n_primitives > MAX_SHAPES_IN_NODE || min_cost < leaf_cost {
                        mid = partition(particles[start..end].iter_mut(), |n| {
                            let mut b: usize = n_buckets
                                * centroid_bounds.offset(&n.bounds_centroid())[split_dimension]
                                    .round() as usize;
                            if b == n_buckets {
                                b = n_buckets - 1;
                            };
                            b <= min_cost_split_bucket
                        }) + start;
                    } else {
                        // println!("leaf");
                        bounds.skip_ptr_or_prim_idx1 = start as u32;
                        bounds.prim_idx2 = end as u32;
                        bounding_circles.push(bounds);
                        return bounding_circles.len() as u32;
                    }
                }
            }

            let curr_idx = bounding_circles.len();
            bounding_circles.push(bounds);

            Bvh::recursive_build(
                bounding_circles,
                particles,
                radius,
                start,
                mid,
                split_method,
            );
            let skip_ptr =
                Bvh::recursive_build(bounding_circles, particles, radius, mid, end, split_method);

            bounding_circles[curr_idx].skip_ptr_or_prim_idx1 = skip_ptr;
        }
        bounding_circles.len() as u32
    }

    pub fn intersect(
        &self,
        particle: &Particle,
        radius: f32,
        g: f32,
        particles: &[Particle],
    ) -> Vec2 {
        let mut ret = Vec2::new(0f32, 0f32);

        let mut idx = 0;
        loop {
            if idx >= self.0.len() {
                break;
            };

            let current_node = &self.0[idx];

            let leaf_node: bool = current_node.prim_idx2 > 0u32;

            if point_in_circle(current_node.centre, current_node.radius, particle.pos) {
                if leaf_node {
                    for prim_idx in (current_node.skip_ptr_or_prim_idx1)..(current_node.prim_idx2) {
                        let p2 = &particles[prim_idx as usize];
                        if point_in_circle(p2.pos, radius, particle.pos) {
                            if std::ptr::eq(particle, p2) {
                                continue;
                            }
                            let d = particle.pos - p2.pos;
                            let r = d.length();
                            if r < f32::EPSILON {
                                continue;
                            }
                            ret += d.normalize_or_zero() / r;
                        }
                    }
                }
                idx += 1;
            } else if leaf_node {
                idx += 1;
            } else {
                idx = current_node.skip_ptr_or_prim_idx1 as usize;
            }
        }
        ret * -g
    }
}

trait Bounded {
    fn bounds(&self, radius: f32) -> NodeInner;
    fn bounds_centroid(&self) -> Vec2;
}

impl Bounded for Particle {
    fn bounds(&self, radius: f32) -> NodeInner {
        NodeInner::new(self.pos, radius, 0u32, 0u32)
    }

    fn bounds_centroid(&self) -> Vec2 {
        self.pos
    }
}

impl AABB {
    pub fn new(first: Vec2, second: Vec2) -> Self {
        AABB { first, second }
    }
    pub fn empty() -> Self {
        AABB {
            first: Vec2::from_array([INFINITY, INFINITY]),
            second: Vec2::from_array([NEG_INFINITY, NEG_INFINITY]),
        }
    }
    pub fn diagonal(&self) -> Vec2 {
        self.second - self.first
    }
    pub fn offset(&self, point: &Vec2) -> Vec2 {
        let mut o = *point - self.first;
        if self.second.x > self.first.x {
            o.x /= self.second.x - self.first.x
        }
        if self.second.y > self.first.y {
            o.y /= self.second.y - self.first.y
        }
        o
    }
    pub fn add_point(&self, point: &Vec2) -> Self {
        // println!("{:?} | {:?}", AABB::new(self.first, self.second), point);
        AABB::new(self.first.min(*point), self.second.max(*point))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_sort() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for power in 0..31 {
            let mut vals = (0..1000)
                .map(|_| rng.gen_range(0..2u32.pow(power)))
                .collect::<Vec<_>>();

            let mut inp: Vec<MortonPrimitive> = vals
                .iter()
                .enumerate()
                .map(|(i, val)| MortonPrimitive {
                    primitive_index: i,
                    morton_code: *val,
                })
                .collect();

            let out = radix_sort(&mut inp)
                .iter()
                .map(|x| x.morton_code)
                .collect::<Vec<_>>();

            vals.sort();
            assert_eq!(out, vals);
        }
    }
}
