use alloc::vec::Vec;

use super::traits::IsMerkleTreeBackend;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn sibling_index(node_index: usize) -> usize {
    if node_index % 2 == 0 {
        node_index - 1
    } else {
        node_index + 1
    }
}

pub fn parent_index(node_index: usize) -> usize {
    if node_index % 2 == 0 {
        (node_index - 1) / 2
    } else {
        node_index / 2
    }
}

// The list of values is completed repeating the last value to a power of two length
pub fn complete_until_power_of_two<T: Clone>(mut values: Vec<T>) -> Vec<T> {
    while !is_power_of_two(values.len()) {
        values.push(values[values.len() - 1].clone());
    }
    values
}

// ! NOTE !
// In this function we say 2^0 = 1 is a power of two.
// In turn, this makes the smallest tree of one leaf, possible.
// The function is private and is only used to ensure the tree
// has a power of 2 number of leaves.
fn is_power_of_two(x: usize) -> bool {
    (x & (x - 1)) == 0
}

// ! CAUTION !
// Make sure n=nodes.len()+1 is a power of two, and the last n/2 elements (leaves) are populated with hashes.
// This function takes no precautions for other cases.
pub fn build<B: IsMerkleTreeBackend>(nodes: &mut [B::Node], leaves_len: usize)
where
    B::Node: Clone,
{
    let mut level_begin_index = leaves_len - 1;
    let mut level_end_index = 2 * level_begin_index;
    while level_begin_index != level_end_index {
        let new_level_begin_index = level_begin_index / 2;
        let new_level_length = level_begin_index - new_level_begin_index;

        let (new_level_iter, children_iter) =
            nodes[new_level_begin_index..level_end_index + 1].split_at_mut(new_level_length);

        #[cfg(feature = "parallel")]
        let parent_and_children_zipped_iter = new_level_iter
            .into_par_iter()
            .zip(children_iter.par_chunks_exact(2));
        #[cfg(not(feature = "parallel"))]
        let parent_and_children_zipped_iter =
            new_level_iter.iter_mut().zip(children_iter.chunks_exact(2));

        parent_and_children_zipped_iter.for_each(|(new_parent, children)| {
            *new_parent = B::hash_new_parent(&children[0], &children[1]);
        });

        level_end_index = level_begin_index - 1;
        level_begin_index = new_level_begin_index;
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    use crate::merkle_tree::{test_merkle::TestBackend, traits::IsMerkleTreeBackend};

    use super::{build, complete_until_power_of_two};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    #[test]
    fn build_merkle_tree_one_element_must_succeed() {
        let mut nodes = [FE::zero()];

        build::<TestBackend<U64PF>>(&mut nodes, 1);
    }

    #[test]
    // expected |2|4|6|8|
    fn hash_leaves_from_a_list_of_field_elemnts() {
        let values: Vec<FE> = (1..5).map(FE::new).collect();
        let hashed_leaves = TestBackend::hash_leaves(&values);
        // New hashing: v+v+1
        // 1+1+1=3, 2+2+1=5, 3+3+1=7, 4+4+1=9
        let list_of_nodes = &[FE::new(3), FE::new(5), FE::new(7), FE::new(9)];
        for (leaf, expected_leaf) in hashed_leaves.iter().zip(list_of_nodes) {
            assert_eq!(leaf, expected_leaf);
        }
    }

    #[test]
    // expected |1|2|3|4|5|5|5|5|
    fn complete_the_length_of_a_list_of_fields_elements_to_be_a_power_of_two() {
        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let hashed_leaves = complete_until_power_of_two(values);

        let mut expected_leaves = (1..6).map(FE::new).collect::<Vec<FE>>();
        expected_leaves.extend([FE::new(5); 3]);

        for (leaf, expected_leaves) in hashed_leaves.iter().zip(expected_leaves) {
            assert_eq!(*leaf, expected_leaves);
        }
    }

    #[test]
    // expected |2| (since 1 is a power of two, length remains 1)
    fn complete_the_length_of_one_field_element_to_be_a_power_of_two() {
        let values: Vec<FE> = vec![FE::new(2)];
        let result_leaves = complete_until_power_of_two(values); // Renamed for clarity

        let expected_leaves = vec![FE::new(2)]; // Expected output is a single element vector
        // The primary assertion is that the length remains 1 because 1 is a power of two.
        assert_eq!(result_leaves.len(), 1);
        // We can also assert the content is as expected.
        assert_eq!(result_leaves, expected_leaves);
    }

    const ROOT: usize = 0;

    #[test]
    // expected |10|10|13|3|7|11|2|1|2|3|4|5|6|7|8|
    fn complete_a_merkle_tree_from_a_set_of_leaves() {
        let leaves: Vec<FE> = (1..9).map(FE::new).collect(); // These are treated as pre-hashed by build
        let leaves_len = leaves.len();

        let mut nodes = vec![FE::zero(); leaves.len() - 1];
        nodes.extend(leaves);

        build::<TestBackend<U64PF>>(&mut nodes, leaves_len);
        // New parent hash: c1+c2+2
        // Leaves: [1,2,3,4,5,6,7,8]
        // P0: 1+2+2=5, 3+4+2=9, 5+6+2=13 (0), 7+8+2=17 (4) -> [5,9,0,4]
        // P1: 5+9+2=16 (3), 0+4+2=6 -> [3,6]
        // R: 3+6+2=11
        assert_eq!(nodes[ROOT], FE::new(11));
    }

    #[test]
    fn build_tree_1_leaf() {
        let leaves_len = 1;
        // nodes = [L0]
        let mut nodes = vec![FE::new(5)]; // Root is the leaf itself
        
        build::<TestBackend<U64PF>>(&mut nodes, leaves_len);
        assert_eq!(nodes.len(), 2 * leaves_len - 1, "Nodes array length mismatch");
        assert_eq!(nodes[ROOT], FE::new(5));
    }

    #[test]
    fn build_tree_2_leaves() {
        let leaves_len = 2;
        // nodes = [P, L0, L1]
        // Leaves start at index leaves_len - 1 = 1
        let l0 = FE::new(10);
        let l1 = FE::new(20);
        let mut nodes = vec![FE::zero(); leaves_len - 1]; // Inner nodes
        nodes.extend(vec![l0, l1]); // Leaf nodes
        
        assert_eq!(nodes.len(), 2 * leaves_len - 1, "Initial nodes array length mismatch"); // Should be 3

        build::<TestBackend<U64PF>>(&mut nodes, leaves_len);
        // New parent hash: l0+l1+2
        // l0=10, l1=20 (7 mod 13)
        // 10+7+2 = 19 (6 mod 13)
        assert_eq!(nodes[ROOT], FE::new(6));
        assert_eq!(nodes[1], l0);
        assert_eq!(nodes[2], l1);
    }

    #[test]
    fn build_tree_4_leaves() {
        let leaves_len = 4;
        // nodes = [R, P01, P23, L0, L1, L2, L3]
        // Leaves start at index leaves_len - 1 = 3
        let l0 = FE::new(1);
        let l1 = FE::new(2);
        let l2 = FE::new(3);
        let l3 = FE::new(4);
        let mut nodes = vec![FE::zero(); leaves_len - 1]; // Inner nodes (3 of them)
        nodes.extend(vec![l0, l1, l2, l3]); // Leaf nodes (4 of them)

        assert_eq!(nodes.len(), 2 * leaves_len - 1, "Initial nodes array length mismatch"); // Should be 7

        build::<TestBackend<U64PF>>(&mut nodes, leaves_len);
        // New parent hash: c1+c2+2
        // Leaves: l0=1, l1=2, l2=3, l3=4
        // P01 = 1+2+2 = 5
        // P23 = 3+4+2 = 9
        // R = P01+P23+2 = 5+9+2 = 16 (3 mod 13)
        assert_eq!(nodes[ROOT], FE::new(3), "Root incorrect");
        assert_eq!(nodes[1], FE::new(5), "Parent P01 incorrect");
        assert_eq!(nodes[2], FE::new(9), "Parent P23 incorrect");
        assert_eq!(nodes[3], l0);
        assert_eq!(nodes[4], l1);
        assert_eq!(nodes[5], l2);
        assert_eq!(nodes[6], l3);
    }

    #[test]
    fn build_tree_3_leaves_padded_to_4() {
        // Mimic behavior of MerkleTree::build -> complete_until_power_of_two -> build
        // Original leaves: 1, 2, 3. Padded leaves for build: 1, 2, 3, 3.
        let leaves_len = 4; // build is called with the padded length

        let l0 = FE::new(1);
        let l1 = FE::new(2);
        let l2 = FE::new(3);
        let l3_padded = FE::new(3); // last element repeated

        let mut nodes = vec![FE::zero(); leaves_len - 1]; // Inner nodes (3 of them)
        nodes.extend(vec![l0, l1, l2, l3_padded]); // Leaf nodes (4 of them)
        
        assert_eq!(nodes.len(), 2 * leaves_len - 1, "Initial nodes array length mismatch"); // Should be 7

        build::<TestBackend<U64PF>>(&mut nodes, leaves_len);
        // New parent hash: c1+c2+2
        // Leaves: l0=1, l1=2, l2=3, l3_padded=3
        // P01 = 1+2+2 = 5
        // P23 = 3+3+2 = 8
        // R = P01+P23+2 = 5+8+2 = 15 (2 mod 13)
        assert_eq!(nodes[ROOT], FE::new(2), "Root incorrect");
        assert_eq!(nodes[1], FE::new(5), "Parent P01 incorrect");
        assert_eq!(nodes[2], FE::new(8), "Parent P23 incorrect");
        assert_eq!(nodes[3], l0);
        assert_eq!(nodes[4], l1);
        assert_eq!(nodes[5], l2);
        assert_eq!(nodes[6], l3_padded);
    }
}
