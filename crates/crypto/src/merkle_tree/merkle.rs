use core::fmt::Display;

use alloc::vec::Vec;

use super::{proof::Proof, traits::IsMerkleTreeBackend, utils::*};

#[derive(Debug)]
pub enum Error {
    OutOfBounds,
}
impl Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Accessed node was out of bound")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// The struct for the Merkle tree, consisting of the root and the nodes.
/// A typical tree would look like this
///                 root
///              /        \
///          leaf 12     leaf 34
///        /         \    /      \
///    leaf 1     leaf 2 leaf 3  leaf 4
/// The bottom leafs correspond to the hashes of the elements, while each upper
/// layer contains the hash of the concatenation of the daughter nodes.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MerkleTree<B: IsMerkleTreeBackend> {
    pub root: B::Node,
    nodes: Vec<B::Node>,
}

const ROOT: usize = 0;

impl<B> MerkleTree<B>
where
    B: IsMerkleTreeBackend,
{
    /// Create a Merkle tree from a slice of data
    pub fn build(unhashed_leaves: &[B::Data]) -> Option<Self> {
        if unhashed_leaves.is_empty() {
            return None;
        }

        let hashed_leaves: Vec<B::Node> = B::hash_leaves(unhashed_leaves);

        //The leaf must be a power of 2 set
        let hashed_leaves = complete_until_power_of_two(hashed_leaves);
        let leaves_len = hashed_leaves.len();

        //The length of leaves minus one inner node in the merkle tree
        //The first elements are overwritten by build function, it doesn't matter what it's there
        let mut nodes = vec![hashed_leaves[0].clone(); leaves_len - 1];
        nodes.extend(hashed_leaves);

        //Build the inner nodes of the tree
        build::<B>(&mut nodes, leaves_len);

        Some(MerkleTree {
            root: nodes[ROOT].clone(),
            nodes,
        })
    }

    /// Returns a Merkle proof for the element/s at position pos
    /// For example, give me an inclusion proof for the 3rd element in the
    /// Merkle tree
    pub fn get_proof_by_pos(&self, pos: usize) -> Option<Proof<B::Node>> {
        let pos = pos + self.nodes.len() / 2;
        let Ok(merkle_path) = self.build_merkle_path(pos) else {
            return None;
        };

        self.create_proof(merkle_path)
    }

    /// Creates a proof from a Merkle pasth
    fn create_proof(&self, merkle_path: Vec<B::Node>) -> Option<Proof<B::Node>> {
        Some(Proof { merkle_path })
    }

    /// Returns the Merkle path for the element/s for the leaf at position pos
    fn build_merkle_path(&self, pos: usize) -> Result<Vec<B::Node>, Error> {
        let mut merkle_path = Vec::new();
        let mut pos = pos;

        while pos != ROOT {
            let Some(node) = self.nodes.get(sibling_index(pos)) else {
                // out of bounds, exit returning the current merkle_path
                return Err(Error::OutOfBounds);
            };
            merkle_path.push(node.clone());

            pos = parent_index(pos);
        }

        Ok(merkle_path)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    use crate::merkle_tree::{merkle::MerkleTree, test_merkle::TestBackend};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    #[test]
    fn build_merkle_tree_from_a_power_of_two_list_of_values() {
        let values: Vec<FE> = (1..5).map(FE::new).collect(); // [1,2,3,4]
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
        // H_L(1)=3, H_L(2)=5, H_L(3)=7, H_L(4)=9. Leaves = [3,5,7,9]
        // P0=H_P(3,5)=3+5+2=10. P1=H_P(7,9)=7+9+2=18 (5). Parents = [10,5]
        // R=H_P(10,5)=10+5+2=17 (4).
        assert_eq!(merkle_tree.root, FE::new(4));
    }

    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn build_merkle_tree_from_an_odd_set_of_leaves() {
        const MODULUS: u64 = 13;
        type U64PF = U64PrimeField<MODULUS>;
        type FE = FieldElement<U64PF>;

        let values: Vec<FE> = (1..6).map(FE::new).collect(); // [1,2,3,4,5]
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
        // H_L(1)=3, H_L(2)=5, H_L(3)=7, H_L(4)=9, H_L(5)=11.
        // Padded hashed leaves (to 8): [3,5,7,9,11,11,11,11]
        // P_L0: H_P(3,5)=10, H_P(7,9)=5, H_P(11,11)=11, H_P(11,11)=11. -> [10,5,11,11]
        // P_L1: H_P(10,5)=4, H_P(11,11)=11. -> [4,11]
        // R: H_P(4,11)=4.
        assert_eq!(merkle_tree.root, FE::new(4));
    }

    #[test]
    fn build_merkle_tree_from_a_single_value() {
        const MODULUS: u64 = 13;
        type U64PF = U64PrimeField<MODULUS>;
        type FE = FieldElement<U64PF>;

        let values: Vec<FE> = vec![FE::new(1)]; // Single element
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
        // H_L(1) = 1+1+1 = 3. This is the root.
        assert_eq!(merkle_tree.root, FE::new(3));
    }

    #[test]
    fn build_empty_tree_should_not_panic() {
        assert!(MerkleTree::<TestBackend<U64PF>>::build(&[]).is_none());
    }

    // Helper function for testing proof generation and verification
    fn run_proof_generation_verification_test(num_actual_leaves: usize) {
        if num_actual_leaves == 0 {
            assert!(MerkleTree::<TestBackend<U64PF>>::build(&[]).is_none());
            return;
        }

        // Create distinct values for leaves to ensure proofs are unique where expected
        let values: Vec<FE> = (1..=num_actual_leaves as u64).map(FE::new).collect();
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

        for leaf_idx in 0..num_actual_leaves {
            let proof = merkle_tree.get_proof_by_pos(leaf_idx).unwrap_or_else(|| {
                panic!(
                    "Failed to get proof for leaf_idx {} with {} actual leaves",
                    leaf_idx, num_actual_leaves
                )
            });
            let original_value = &values[leaf_idx]; // This is B::Data

            // Verify the proof
            // The `index` in `verify` is the original 0-based index of the leaf
            let is_valid = proof.verify::<TestBackend<U64PF>>(
                &merkle_tree.root,
                leaf_idx,
                original_value,
            );
            assert!(
                is_valid,
                "Proof verification failed for leaf_idx {} with {} actual leaves. Root: {:?}, Proof: {:?}",
                leaf_idx,
                num_actual_leaves,
                merkle_tree.root,
                proof.merkle_path
            );
        }
    }

    #[test]
    fn proof_generation_verification_1_leaf() {
        run_proof_generation_verification_test(1);
    }

    #[test]
    fn proof_generation_verification_2_leaves() {
        run_proof_generation_verification_test(2);
    }

    #[test]
    fn proof_generation_verification_3_leaves() {
        // Padded to 4 leaves internally.
        // Original values [1,2,3]. Modulo 13.
        // Hashed leaves (v+v+1): H_L(1)=3, H_L(2)=5, H_L(3)=7.
        // Padded hashed leaves (to 4): [3,5,7,7].
        // Parents L0 (c1+c2+2):
        //   P0 = H_P(3,5) = 3+5+2 = 10
        //   P1 = H_P(7,7) = 7+7+2 = 16 (3 mod 13)
        // Root (c1+c2+2):
        //   R = H_P(10,3) = 10+3+2 = 15 (2 mod 13)
        run_proof_generation_verification_test(3);
    }

    #[test]
    fn proof_generation_verification_4_leaves() {
        run_proof_generation_verification_test(4);
    }

    #[test]
    fn proof_generation_verification_5_leaves() {
        // Padded to 8 leaves internally.
        run_proof_generation_verification_test(5);
    }

    #[test]
    fn proof_generation_verification_7_leaves() {
        run_proof_generation_verification_test(7);
    }

    #[test]
    fn proof_generation_verification_8_leaves() {
        run_proof_generation_verification_test(8);
    }

    #[test]
    fn get_proof_for_out_of_bounds_index() {
        let values: Vec<FE> = (1..=4).map(FE::new).collect();
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
        
        // `get_proof_by_pos` takes the original leaf index.
        // The `MerkleTree::build` function itself doesn't store the original unhashed leaves count directly,
        // but relies on the caller of `get_proof_by_pos` to use a valid original index.
        // The check `if pos >= self.data.len()` that was mentioned in some problem descriptions
        // isn't present in the provided code for `get_proof_by_pos`.
        // Instead, `get_proof_by_pos` calculates an internal index and `build_merkle_path`
        // might return an error if this index leads to an out-of-bounds access on `self.nodes`.
        // Let's test an index that, after transformation, would be out of bounds for the leaves
        // or would lead to an invalid path construction.

        // If `leaves_len` (padded) is 4. `nodes.len()` is 7. `nodes.len()/2` is 3.
        // `pos = external_pos + 3`.
        // If `external_pos = 0`, internal `pos = 3`.
        // If `external_pos = 3`, internal `pos = 6`.
        // If `external_pos = 4` (out of bounds for original 4 leaves):
        // Internal `pos = 4 + 3 = 7`. `self.nodes[7]` would be an error.
        // `build_merkle_path` uses `self.nodes.get(sibling_index(pos))`.
        // If pos=7, sibling_index(7) = 6. self.nodes.get(6) is valid.
        // parent_index(7) = 3.
        // Next iter: pos=3. sibling_index(3) = 2. self.nodes.get(2) is valid.
        // parent_index(3) = 1.
        // Next iter: pos=1. sibling_index(1) = 0. self.nodes.get(0) is valid.
        // parent_index(1) = 0.
        // Next iter: pos=0 (ROOT). Loop terminates.
        // This implies that `get_proof_by_pos` might return a "valid" proof for an out-of-bounds original index
        // if that index, once mapped, corresponds to an actual node in the tree, and its path to root can be built.
        // However, this "proof" would not verify against the intended original value at the out-of-bounds position.
        // The current `get_proof_by_pos` doesn't have a direct check against original leaves count.
        // The existing tests for MerkleTree (e.g. in `proof.rs` like `create_a_proof_over_value_that_belongs_to_a_given_merkle_tree_when_given_the_leaf_position`)
        // use `merkle_tree.get_proof_by_pos(1).unwrap()` where 1 is a valid index.
        // The `merkle_tree` object itself doesn't store the original `unhashed_leaves` or their count.
        // This means `get_proof_by_pos` *cannot* directly check `pos >= original_leaves_count`.
        // It's up to the caller to provide a valid `pos`.
        // The current implementation of `get_proof_by_pos` will return `None` if `build_merkle_path` returns `Err`,
        // which happens if `self.nodes.get(sibling_index(pos))` is `None`.
        // For a tree with 4 leaves (padded to 4), `leaves_len = 4`. `nodes.len() = 7`.
        // `get_proof_by_pos(pos_orig)` computes internal `pos_internal = pos_orig + 3`.
        // If `pos_orig = 4`. `pos_internal = 7`. `self.nodes` has indices 0..6. So `self.nodes.get(anything >= 7)` is `None`.
        // `sibling_index(7)` is 6. `self.nodes.get(6)` is valid. Path continues.
        // This indicates that `get_proof_by_pos` might not return `None` for an out-of-bounds `pos_orig` if `pos_internal` is valid.
        // Let's re-evaluate: `pos` in `build_merkle_path` refers to an index in `self.nodes`.
        // `sibling_index(pos)` could be out of bounds if `pos` is 0 and `sibling_index(0)` is -1 (usize wraps).
        // `sibling_index` is: `if node_index % 2 == 0 { node_index - 1 } else { node_index + 1 }`
        // This is for a 1-indexed system usually.
        // The `parent_index` and `sibling_index` in `utils.rs` are:
        // sibling_index(node_idx): if node_idx is even, returns node_idx-1. if odd, node_idx+1. (These are relative to parent's children block)
        // parent_index(node_idx): if node_idx is even, (node_idx-1)/2. if odd, node_idx/2.
        // These utils are for the flat array representation where root is 0, its children are 1,2 etc.
        //   0
        //  1  2
        // 3 4 5 6
        // If `pos_internal = 7` (which is out of bounds for `nodes` of len 7).
        // `self.nodes.get(sibling_index(7))` will be `self.nodes.get(6)`.
        // The first check in `build_merkle_path` is `while pos != ROOT`. If `pos` starts out of bounds but high,
        // the `self.nodes.get` will eventually catch it.
        // If `pos_orig = 4`, `leaves_len = 4` (padded). `nodes.len() = 7`. `nodes.len()/2 = 3`.
        // `pos_internal = 4 + 3 = 7`.
        // `build_merkle_path(7)`:
        //   `pos = 7`. `7 != ROOT (0)`.
        //   `sibling_index(7)` is `8` (since 7 is odd, 7+1). `self.nodes.get(8)` is `None`. Returns `Err(OutOfBounds)`.
        //   So `get_proof_by_pos(4)` will correctly return `None`.

        assert!(merkle_tree.get_proof_by_pos(num_actual_leaves).is_none(), "Proof should be None for out-of-bounds index");
        assert!(merkle_tree.get_proof_by_pos(num_actual_leaves + 1).is_none(), "Proof should be None for further out-of-bounds index");
    }

     #[test]
    fn proof_generation_verification_10_leaves() {
        // Padded to 16 leaves internally.
        run_proof_generation_verification_test(10);
    }
}
