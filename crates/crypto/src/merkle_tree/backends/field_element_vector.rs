use core::marker::PhantomData;

use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use alloc::vec::Vec;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};
use sha3::{
    digest::{generic_array::GenericArray, OutputSizeUser},
    Digest,
};

/// A Merkle tree backend that uses a generic `Digest` (e.g., SHA-256, Keccak256)
/// to hash `Vec<FieldElement<F>>` data for leaves. Internal nodes (parents) are
/// formed by hashing the concatenated byte representations of their children.
/// The output node is a fixed-size byte array.
///
/// **Security Considerations - Domain Separation:**
/// The current implementation of `FieldElementVectorBackend`:
/// - For leaves (`hash_data`): Iterates through the input `Vec<FieldElement<F>>`,
///   serializes each element to bytes, and feeds these sequentially to the hasher.
/// - For internal nodes (`hash_new_parent`): Concatenates and hashes the byte
///   representations of child nodes.
///
/// ```
/// // Simplified conceptual representation:
/// // hash_data(input: &Vec<FE>) -> H(FE_0.as_bytes() || FE_1.as_bytes() || ...)
/// // hash_new_parent(left: &Node, right: &Node) -> H(left_bytes || right_bytes)
/// ```
///
/// This does **not** inherently provide domain separation between leaf data and internal
/// node construction, nor does it distinguish between different leaf data structures if, for
/// example, `[FE(1), FE(23)]` and `[FE(12), FE(3)]` could serialize to the same byte sequence.
/// It is crucial to ensure that the hash of a leaf cannot collide with the hash of an
/// internal node, and that distinct leaf data structures hash to distinct values.
///
/// **Recommendation for Secure Implementation:**
/// For production use, this backend should be modified, or a new backend created,
/// to incorporate domain separation. This typically involves prepending a unique domain tag
/// (e.g., a specific byte or sequence of bytes) to the data before hashing:
/// - For leaves: `H(LEAF_DOMAIN_TAG || FE_0.as_bytes() || FE_1.as_bytes() || ...)`
/// - For internal nodes: `H(NODE_DOMAIN_TAG || left_bytes || right_bytes)`
///
/// Where `LEAF_DOMAIN_TAG` and `NODE_DOMAIN_TAG` are distinct constant byte sequences.
/// This ensures that the inputs to the hash function for leaves and internal nodes
/// always differ, preventing type confusion attacks.
#[derive(Clone)]
pub struct FieldElementVectorBackend<F, D: Digest, const NUM_BYTES: usize> {
    phantom1: PhantomData<F>,
    phantom2: PhantomData<D>,
}

impl<F, D: Digest, const NUM_BYTES: usize> Default for FieldElementVectorBackend<F, D, NUM_BYTES> {
    fn default() -> Self {
        Self {
            phantom1: PhantomData,
            phantom2: PhantomData,
        }
    }
}

impl<F, D: Digest, const NUM_BYTES: usize> IsMerkleTreeBackend
    for FieldElementVectorBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    [u8; NUM_BYTES]: From<GenericArray<u8, <D as OutputSizeUser>::OutputSize>>,
    Vec<FieldElement<F>>: Sync + Send,
{
    type Node = [u8; NUM_BYTES];
    type Data = Vec<FieldElement<F>>;

    fn hash_data(input: &Vec<FieldElement<F>>) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        for element in input.iter() {
            hasher.update(element.as_bytes());
        }
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        let mut result_hash = [0_u8; NUM_BYTES];
        result_hash.copy_from_slice(&hasher.finalize());
        result_hash
    }
}

/// A Merkle tree backend that uses a `Poseidon` hash function.
/// `Data` is `Vec<FieldElement<P::F>>` (hashed with `P::hash_many` for leaves).
/// `Node` is `FieldElement<P::F>` (internal nodes hashed with `P::hash`).
///
/// **Security Considerations - Domain Separation:**
/// The current implementation of `BatchPoseidonTree` uses:
/// - `P::hash_many(input_vector)` for leaves.
/// - `P::hash(left_child, right_child)` for internal nodes.
///
/// ```
/// // Simplified conceptual representation:
/// // hash_data(input: &Vec<FE>) -> P::hash_many(input)
/// // hash_new_parent(left: &FE, right: &FE) -> P::hash(left, right)
/// ```
///
/// While `P::hash_many` and `P::hash` are different functions and might use different
/// internal configurations (e.g. different round constants based on input arity),
/// making explicit domain separation a part of the input fed to the Poseidon permutation
/// is a stronger security practice. This ensures that inputs for leaf construction are
/// unambiguously distinct from inputs for internal node construction.
///
/// **Recommendation for Secure Implementation:**
/// For production use, consider modifying this backend or creating a new one
/// to incorporate explicit domain separation. This can be achieved by dedicating one of the
/// input field elements as a domain tag when calling the Poseidon hash function:
/// - For leaves: Prepend a `LEAF_DOMAIN_TAG_FE` to the `input` vector before calling `P::hash_many`.
///   Example: `let tagged_input = [vec![LEAF_DOMAIN_TAG_FE], input.as_slice()].concat(); P::hash_many(&tagged_input)`
/// - For internal nodes: `P::hash(&[NODE_DOMAIN_TAG_FE, left, right])` (if `P::hash` can take 3 inputs,
///   otherwise adjust based on the Poseidon interface, e.g., by hashing `NODE_DOMAIN_TAG_FE` with `left` first, then with `right`).
///
/// Where `LEAF_DOMAIN_TAG_FE` and `NODE_DOMAIN_TAG_FE` are distinct, constant field elements.
/// This makes the inputs to the Poseidon permutation unambiguously different for leaves versus internal nodes.
/// Alternatively, if the Poseidon implementation supports it, using different permutation variants
/// or initial states for leaves and nodes could also achieve domain separation.
#[derive(Clone, Default)]
pub struct BatchPoseidonTree<P: Poseidon + Default> {
    _poseidon: PhantomData<P>,
}

impl<P> IsMerkleTreeBackend for BatchPoseidonTree<P>
where
    P: Poseidon + Default,
    Vec<FieldElement<P::F>>: Sync + Send,
    FieldElement<P::F>: Sync + Send,
{
    type Node = FieldElement<P::F>;
    type Data = Vec<FieldElement<P::F>>;

    fn hash_data(input: &Vec<FieldElement<P::F>>) -> FieldElement<P::F> {
        P::hash_many(input)
    }

    fn hash_new_parent(
        left: &FieldElement<P::F>,
        right: &FieldElement<P::F>,
    ) -> FieldElement<P::F> {
        P::hash(left, right)
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };
    use sha2::Sha512;
    use sha3::{Keccak256, Keccak512, Sha3_256, Sha3_512};

    use crate::merkle_tree::{
        backends::field_element_vector::FieldElementVectorBackend, merkle::MerkleTree,
    };

    type F = Stark252PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn hash_data_field_element_backend_works_with_sha3_256() {
        let values = [
            vec![FE::from(2u64), FE::from(11u64)],
            vec![FE::from(3u64), FE::from(14u64)],
            vec![FE::from(4u64), FE::from(7u64)],
            vec![FE::from(5u64), FE::from(3u64)],
            vec![FE::from(6u64), FE::from(5u64)],
            vec![FE::from(7u64), FE::from(16u64)],
            vec![FE::from(8u64), FE::from(19u64)],
            vec![FE::from(9u64), FE::from(21u64)],
        ];
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Sha3_256, 32>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Sha3_256, 32>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_keccak256() {
        let values = [
            vec![FE::from(2u64), FE::from(11u64)],
            vec![FE::from(3u64), FE::from(14u64)],
            vec![FE::from(4u64), FE::from(7u64)],
            vec![FE::from(5u64), FE::from(3u64)],
            vec![FE::from(6u64), FE::from(5u64)],
            vec![FE::from(7u64), FE::from(16u64)],
            vec![FE::from(8u64), FE::from(19u64)],
            vec![FE::from(9u64), FE::from(21u64)],
        ];
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Keccak256, 32>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Keccak256, 32>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_sha3_512() {
        let values = [
            vec![FE::from(2u64), FE::from(11u64)],
            vec![FE::from(3u64), FE::from(14u64)],
            vec![FE::from(4u64), FE::from(7u64)],
            vec![FE::from(5u64), FE::from(3u64)],
            vec![FE::from(6u64), FE::from(5u64)],
            vec![FE::from(7u64), FE::from(16u64)],
            vec![FE::from(8u64), FE::from(19u64)],
            vec![FE::from(9u64), FE::from(21u64)],
        ];
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Sha3_512, 64>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Sha3_512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_keccak512() {
        let values = [
            vec![FE::from(2u64), FE::from(11u64)],
            vec![FE::from(3u64), FE::from(14u64)],
            vec![FE::from(4u64), FE::from(7u64)],
            vec![FE::from(5u64), FE::from(3u64)],
            vec![FE::from(6u64), FE::from(5u64)],
            vec![FE::from(7u64), FE::from(16u64)],
            vec![FE::from(8u64), FE::from(19u64)],
            vec![FE::from(9u64), FE::from(21u64)],
        ];
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Keccak512, 64>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Keccak512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_sha2_512() {
        let values = [
            vec![FE::from(2u64), FE::from(11u64)],
            vec![FE::from(3u64), FE::from(14u64)],
            vec![FE::from(4u64), FE::from(7u64)],
            vec![FE::from(5u64), FE::from(3u64)],
            vec![FE::from(6u64), FE::from(5u64)],
            vec![FE::from(7u64), FE::from(16u64)],
            vec![FE::from(8u64), FE::from(19u64)],
            vec![FE::from(9u64), FE::from(21u64)],
        ];
        let merkle_tree =
            MerkleTree::<FieldElementVectorBackend<F, Sha512, 64>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementVectorBackend<F, Sha512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }
}
