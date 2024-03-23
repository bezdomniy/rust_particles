use glam::Vec2;
use itertools::partition;
use std::f32::{consts::PI, INFINITY, NEG_INFINITY};

use crate::app::Particle;

static MAX_SHAPES_IN_NODE: usize = 4;

#[derive(Debug, Default, Copy, Clone)]
pub struct NodeInner {
    pub centre: Vec2,
    pub radius: f32,
    _padding0: u32,
    pub skip_ptr_or_prim_idx1: u32,
    pub prim_idx2: u32,
    _padding1: u32,
    _padding2: u32,
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

impl NodeInner {
    pub fn new(centre: Vec2, radius: f32, skip_ptr_or_prim_idx1: u32, prim_idx2: u32) -> Self {
        NodeInner {
            centre: centre,
            radius: radius,
            _padding0: 0u32,
            skip_ptr_or_prim_idx1,
            prim_idx2,
            _padding1: 0u32,
            _padding2: 0u32,
        }
    }

    pub fn empty() -> Self {
        NodeInner {
            centre: Vec2::new(f32::INFINITY, f32::INFINITY),
            radius: 0f32,
            _padding0: 0u32,
            skip_ptr_or_prim_idx1: 0u32,
            prim_idx2: 0u32,
            _padding1: 0u32,
            _padding2: 0u32,
        }
    }

    pub fn merge(&self, other: &NodeInner) -> Self {
        let d = other.centre.distance(self.centre);

        if d + self.radius <= other.radius {
            return *other;
        }
        if d + other.radius <= self.radius {
            return *self;
        }

        let new_centre = (self.centre + other.centre) / 2f32;
        let new_radius = f32::max(self.radius, other.radius) + d / 2f32;

        NodeInner::new(
            new_centre,
            new_radius,
            other.skip_ptr_or_prim_idx1,
            other.prim_idx2,
        )
    }

    pub fn surface_area(&self) -> f32 {
        (self.radius * self.radius) * PI
    }
}

// #[derive(Debug, Default)]
// #[repr(C)]
// pub struct Bvh {
//     pub inner_nodes: Vec<NodeInner>,
// }

// type Bvh = Vec<NodeInner>;
pub struct Bvh(Vec<NodeInner>);

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum SplitMethod {
    Middle,
    EqualCounts,
    Sah,
}

impl Bvh {
    pub fn empty() -> Self {
        Bvh(vec![])
    }

    pub fn new(particles: &mut [Particle], radius: f32) -> Self {
        let object_inner_nodes = Bvh::build(particles, radius);

        // object_leaf_nodes.push(triangles);

        if object_inner_nodes.is_empty() {
            return Bvh::empty();
        }

        // let n_objects = object_inner_nodes.len() as u32;

        Bvh(object_inner_nodes)
    }

    fn build(primitives: &mut [Particle], radius: f32) -> Vec<NodeInner> {
        let mut bounding_circles: Vec<NodeInner> =
            Vec::with_capacity(primitives.len().next_power_of_two());

        let split_method = SplitMethod::EqualCounts;

        Bvh::recursive_build(
            &mut bounding_circles,
            primitives,
            radius,
            0,
            primitives.len(),
            split_method,
        );

        // println!("{}", bounding_circles.len());
        bounding_circles
    }

    // TODO: make this return the skip pointer so it can bubble up
    fn recursive_build(
        bounding_circles: &mut Vec<NodeInner>,
        primitives: &mut [Particle],
        radius: f32,
        start: usize,
        end: usize,
        split_method: SplitMethod,
    ) -> u32 {
        // for x in &primitives[start..end] {
        //     println!("{}, {}, {:?}", start, end, x);
        // }
        // println!("start end: {:?} {:?}", start, end);
        let centroid_bounds = primitives[start..end]
            .iter()
            .fold(AABB::empty(), |acc, new| {
                acc.add_point(&new.bounds_centroid())
            });

        let mut bounds = primitives[start..end]
            .iter()
            .fold(NodeInner::empty(), |acc, new| {
                acc.merge(&new.bounds(radius))
            });

        // println!("cb: {:?}", centroid_bounds);

        let diagonal = centroid_bounds.diagonal();

        let split_dimension = if diagonal.x > diagonal.y { 0 } else { 1 };

        let n_shapes = end - start;
        // let mid = (start + end) / 2;

        let is_leaf: bool = centroid_bounds.first[split_dimension]
            == centroid_bounds.second[split_dimension]
            || n_shapes <= 2;

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

                mid = partition(primitives[start..end].iter_mut(), |n| {
                    n.bounds_centroid()[split_dimension] < pmid
                }) + start;

                if mid != start && mid != end {
                    fallthrough = true;
                }
            }

            if fallthrough || matches!(split_method, SplitMethod::EqualCounts) {
                mid = (start + end) / 2;
                primitives[start..end].select_nth_unstable_by(mid - start, |a, b| {
                    a.bounds_centroid()[split_dimension]
                        .partial_cmp(&b.bounds_centroid()[split_dimension])
                        .unwrap()
                });
            }

            if matches!(split_method, SplitMethod::Sah) {
                // println!("{:?}", centroid_bounds);
                if n_shapes <= 2 {
                    mid = (start + end) / 2;
                    primitives[start..end].select_nth_unstable_by(mid - start, |a, b| {
                        a.bounds_centroid()[split_dimension]
                            .partial_cmp(&b.bounds_centroid()[split_dimension])
                            .unwrap()
                    });
                } else {
                    let n_buckets: usize = 12;
                    let mut buckets = vec![NodeInner::empty(); n_buckets];

                    for triangle in primitives.iter().take(end).skip(start) {
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

                    let leaf_cost = n_shapes as f32;
                    if n_shapes > MAX_SHAPES_IN_NODE || min_cost < leaf_cost {
                        mid = partition(primitives[start..end].iter_mut(), |n| {
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
                primitives,
                radius,
                start,
                mid,
                split_method,
            );
            let skip_ptr =
                Bvh::recursive_build(bounding_circles, primitives, radius, mid, end, split_method);

            bounding_circles[curr_idx].skip_ptr_or_prim_idx1 = skip_ptr;

            // println!("{:?}", bounds);

            // bounds.skip_ptr_or_prim_idx1 = 2u32.pow((bvh_height - level) as u32) - 1;
            // bounds.skip_ptr_or_prim_idx1 = 1;
        }
        bounding_circles.len() as u32
    }

    pub fn intersect(
        &self,
        particle: &Particle,
        radius: f32,
        particles: &[Particle],
    ) -> Vec<Particle> {
        let mut ret = vec![];

        let mut idx = 0;
        loop {
            if idx >= self.0.len() {
                break;
            };

            let current_node: NodeInner = self.0[idx];

            let leaf_node: bool = current_node.prim_idx2 > 0u32;

            if point_in_circle(current_node.centre, current_node.radius, particle.pos) {
                if leaf_node {
                    for prim_idx in (current_node.skip_ptr_or_prim_idx1)..(current_node.prim_idx2) {
                        //TODO pass radius
                        if point_in_circle(particles[prim_idx as usize].pos, radius, particle.pos) {
                            ret.push(particle.clone());
                        }

                        // let next_intersection = intersectTriangle(ray, primIdx, ret, object_id);

                        // if ((next_intersection.closestT < inIntersection.closestT)
                        //     && (next_intersection.closestT > EPSILON))
                        // {
                        //     ret = next_intersection;
                        // }
                    }
                }
                idx += 1;
            } else if leaf_node {
                idx += 1;
            } else {
                idx = current_node.skip_ptr_or_prim_idx1 as usize;
            }
        }
        return ret;
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
        // println!("{:?}", self.pos);
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
