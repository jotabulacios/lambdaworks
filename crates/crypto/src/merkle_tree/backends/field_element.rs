use crate::hash::poseidon::Poseidon;

use crate::merkle_tree::traits::IsMerkleTreeBackend;
use core::marker::PhantomData;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};
use sha3::{
    digest::{generic_array::GenericArray, OutputSizeUser},
    Digest,
};

/// A Merkle tree backend that uses a generic `Digest` (e.g., SHA-256, Keccak256)
/// to hash `FieldElement` data. The output node is a fixed-size byte array.
///
/// **Security Considerations - Domain Separation:**
/// The current implementation of `FieldElementBackend` directly hashes the byte representation
/// of the field element for leaves, and concatenates and hashes the byte representations
/// of child nodes for internal nodes.
///
/// ```
/// // Simplified conceptual representation:
/// // hash_data(input: &FE) -> H(input.as_bytes())
/// // hash_new_parent(left: &Node, right: &Node) -> H(left_bytes || right_bytes)
/// ```
///
/// This does **not** inherently provide domain separation between leaf data and internal
/// node construction. In a security-critical application, it is crucial to ensure that
/// the hash of a leaf cannot collide with the hash of an internal node, even if the
/// underlying byte sequences were maliciously crafted.
///
/// **Recommendation for Secure Implementation:**
/// For production use, this backend should be modified, or a new backend created,
/// to incorporate domain separation. This typically involves prepending a unique domain tag
/// (e.g., a specific byte or sequence of bytes) to the data before hashing:
/// - For leaves: `H(LEAF_DOMAIN_TAG || input.as_bytes())`
/// - For internal nodes: `H(NODE_DOMAIN_TAG || left_bytes || right_bytes)`
///
/// Where `LEAF_DOMAIN_TAG` and `NODE_DOMAIN_TAG` are distinct constant byte sequences.
/// This ensures that the inputs to the hash function for leaves and internal nodes
/// always differ, preventing type confusion attacks.
#[derive(Clone)]
pub struct FieldElementBackend<F, D: Digest, const NUM_BYTES: usize> {
    phantom1: PhantomData<F>,
    phantom2: PhantomData<D>,
}

impl<F, D: Digest, const NUM_BYTES: usize> Default for FieldElementBackend<F, D, NUM_BYTES> {
    fn default() -> Self {
        Self {
            phantom1: PhantomData,
            phantom2: PhantomData,
        }
    }
}

impl<F, D: Digest, const NUM_BYTES: usize> IsMerkleTreeBackend
    for FieldElementBackend<F, D, NUM_BYTES>
where
    F: IsField,
    FieldElement<F>: AsBytes + Sync + Send,
    [u8; NUM_BYTES]: From<GenericArray<u8, <D as OutputSizeUser>::OutputSize>>,
{
    type Node = [u8; NUM_BYTES];
    type Data = FieldElement<F>;

    fn hash_data(input: &FieldElement<F>) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(input.as_bytes());
        hasher.finalize().into()
    }

    fn hash_new_parent(left: &[u8; NUM_BYTES], right: &[u8; NUM_BYTES]) -> [u8; NUM_BYTES] {
        let mut hasher = D::new();
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().into()
    }
}

/// A Merkle tree backend that uses a `Poseidon` hash function.
/// Both `Data` and `Node` types are `FieldElement<P::F>`.
///
/// **Security Considerations - Domain Separation:**
/// The current implementation of `TreePoseidon` directly calls the underlying
/// Poseidon hash functions:
///
/// ```
/// // Simplified conceptual representation:
/// // hash_data(input: &FE) -> P::hash_single(input)
/// // hash_new_parent(left: &FE, right: &FE) -> P::hash(left, right)
/// ```
///
/// While `P::hash_single` and `P::hash` might use different numbers of inputs or
/// internal configurations (e.g. different round constants based on input arity),
/// explicit domain separation is a stronger security practice. It ensures that the input
/// to the Poseidon function for a leaf hash is clearly distinguished from the inputs
/// for an internal node hash.
///
/// **Recommendation for Secure Implementation:**
/// For production use, consider modifying this backend or creating a new one
/// to incorporate explicit domain separation. When using Poseidon (or other
/// arithmetization-oriented hashes), this can be achieved by dedicating one of the
/// input field elements as a domain tag:
/// - For leaves: `P::hash(&[LEAF_DOMAIN_TAG_FE, input])` (assuming `P::hash` can take variable inputs or padding)
///   or modify `P::hash_single` if it's specifically for one element, e.g. by passing a domain parameter.
/// - For internal nodes: `P::hash(&[NODE_DOMAIN_TAG_FE, left, right])`
///
/// Where `LEAF_DOMAIN_TAG_FE` and `NODE_DOMAIN_TAG_FE` are distinct, constant field elements.
/// This makes the inputs to the Poseidon permutation unambiguously different for leaves versus internal nodes.
/// Alternatively, if the Poseidon implementation supports it, using different permutation variants
/// or initial states for leaves and nodes could also achieve domain separation.
#[derive(Clone, Default)]
pub struct TreePoseidon<P: Poseidon + Default> {
    _poseidon: PhantomData<P>,
}

impl<P> IsMerkleTreeBackend for TreePoseidon<P>
where
    P: Poseidon + Default,
    FieldElement<P::F>: Sync + Send,
{
    type Node = FieldElement<P::F>;
    type Data = FieldElement<P::F>;

    fn hash_data(input: &FieldElement<P::F>) -> FieldElement<P::F> {
        P::hash_single(input)
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
    use alloc::vec::Vec;
    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };
    use sha3::{Keccak256, Keccak512, Sha3_256, Sha3_512};

    use crate::merkle_tree::{backends::field_element::FieldElementBackend, merkle::MerkleTree};

    type F = Stark252PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn hash_data_field_element_backend_works_with_keccak_256() {
        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree =
            MerkleTree::<FieldElementBackend<F, Keccak256, 32>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementBackend<F, Keccak256, 32>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_sha3_256() {
        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree =
            MerkleTree::<FieldElementBackend<F, Sha3_256, 32>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementBackend<F, Sha3_256, 32>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_keccak_512() {
        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree =
            MerkleTree::<FieldElementBackend<F, Keccak512, 64>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementBackend<F, Keccak512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }

    #[test]
    fn hash_data_field_element_backend_works_with_sha3_512() {
        let values: Vec<FE> = (1..6).map(FE::from).collect();
        let merkle_tree =
            MerkleTree::<FieldElementBackend<F, Sha3_512, 64>>::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(proof.verify::<FieldElementBackend<F, Sha3_512, 64>>(
            &merkle_tree.root,
            0,
            &values[0]
        ));
    }
}
