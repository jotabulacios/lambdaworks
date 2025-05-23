use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use lambdaworks_math::traits::Serializable;
use lambdaworks_math::{errors::DeserializationError, traits::Deserializable};

use super::traits::IsMerkleTreeBackend;

/// Stores a merkle path to some leaf.
/// Internally, the necessary hashes are stored from root to leaf in the
/// `merkle_path` field, in such a way that, if the merkle tree is of height `n`, the
/// `i`-th element of `merkle_path` is the sibling node in the `n - 1 - i`-th check
/// when verifying.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Proof<T: PartialEq + Eq> {
    pub merkle_path: Vec<T>,
}

impl<T: PartialEq + Eq> Proof<T> {
    /// Verifies a Merkle inclusion proof for the value contained at leaf index.
    pub fn verify<B>(&self, root_hash: &B::Node, mut index: usize, value: &B::Data) -> bool
    where
        B: IsMerkleTreeBackend<Node = T>,
    {
        let mut hashed_value = B::hash_data(value);

        for sibling_node in self.merkle_path.iter() {
            if index % 2 == 0 {
                hashed_value = B::hash_new_parent(&hashed_value, sibling_node);
            } else {
                hashed_value = B::hash_new_parent(sibling_node, &hashed_value);
            }

            index >>= 1;
        }

        root_hash == &hashed_value
    }
}

#[cfg(feature = "alloc")]
impl<T> Serializable for Proof<T>
where
    T: Serializable + PartialEq + Eq,
{
    fn serialize(&self) -> Vec<u8> {
        let mut serialized_proof = Vec::new();
        for node in &self.merkle_path {
            let node_bytes = node.serialize();
            let len = node_bytes.len() as u32; // Assuming node length fits in u32
            serialized_proof.extend_from_slice(&len.to_be_bytes());
            serialized_proof.extend_from_slice(&node_bytes);
        }
        serialized_proof
    }
}

impl<T> Deserializable for Proof<T>
where
    T: Deserializable + PartialEq + Eq,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        let mut merkle_path = Vec::new();
        let mut current_offset = 0;
        while current_offset < bytes.len() {
            // Read length (u32, 4 bytes)
            if current_offset + 4 > bytes.len() {
                return Err(DeserializationError::InvalidAmountOfBytes);
            }
            let len_bytes: [u8; 4] = bytes[current_offset..current_offset + 4]
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?; // Should not fail due to check above
            let len = u32::from_be_bytes(len_bytes) as usize;
            current_offset += 4;

            // Read node data
            if current_offset + len > bytes.len() {
                return Err(DeserializationError::InvalidAmountOfBytes);
            }
            let node_bytes = &bytes[current_offset..current_offset + len];
            let node = T::deserialize(node_bytes)?;
            merkle_path.push(node);
            current_offset += len;
        }

        if current_offset != bytes.len() {
            // This case should ideally be caught by the length checks if data is malformed,
            // but as a safeguard:
            return Err(DeserializationError::TrailingBytes);
        }

        Ok(Self { merkle_path })
    }
}
#[cfg(test)]
mod tests {

    #[cfg(feature = "alloc")]
    use super::Proof;
    use alloc::vec::Vec;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};
    #[cfg(feature = "alloc")]
    use lambdaworks_math::traits::{Deserializable, Serializable};
    #[cfg(feature = "alloc")]
    use lambdaworks_math::errors::DeserializationError; // For VariableLengthNode test

    use crate::merkle_tree::{merkle::MerkleTree, test_merkle::TestBackend};

    /// Small field useful for starks, sometimes called min i goldilocks
    /// Used in miden and winterfell
    // This field shouldn't be defined inside the merkle tree module
    pub type Ecgfp5 = U64PrimeField<0xFFFF_FFFF_0000_0001_u64>;
    pub type Ecgfp5FE = FieldElement<Ecgfp5>;
    pub type TestMerkleTreeEcgfp = MerkleTree<TestBackend<Ecgfp5>>;
    #[cfg(feature = "alloc")]
    pub type TestProofEcgfp5 = Proof<Ecgfp5FE>;

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    #[test]
    #[cfg(feature = "alloc")]
    fn serialize_proof_and_deserialize_using_be_it_get_a_consistent_proof() {
        let merkle_path = [Ecgfp5FE::new(2), Ecgfp5FE::new(1), Ecgfp5FE::new(1)].to_vec();
        let original_proof = TestProofEcgfp5 { merkle_path };
        let serialize_proof = original_proof.serialize();
        let proof: TestProofEcgfp5 = Proof::deserialize(&serialize_proof).unwrap();

        for (o_node, node) in original_proof.merkle_path.iter().zip(proof.merkle_path) {
            assert_eq!(*o_node, node);
        }
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn serialize_proof_and_deserialize_using_le_it_get_a_consistent_proof() {
        let merkle_path = [Ecgfp5FE::new(2), Ecgfp5FE::new(1), Ecgfp5FE::new(1)].to_vec();
        let original_proof = TestProofEcgfp5 { merkle_path };
        let serialize_proof = original_proof.serialize();
        let proof: TestProofEcgfp5 = Proof::deserialize(&serialize_proof).unwrap();

        for (o_node, node) in original_proof.merkle_path.iter().zip(proof.merkle_path) {
            assert_eq!(*o_node, node);
        }
    }

    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn create_a_proof_over_value_that_belongs_to_a_given_merkle_tree_when_given_the_leaf_position()
    {
        let values: Vec<FE> = (1..6).map(FE::new).collect(); // [1,2,3,4,5]
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
        let proof = &merkle_tree.get_proof_by_pos(1).unwrap(); // Proof for value FE(2) at original index 1

        // Recalculation with new hashing (HL: v+v+1, HP: c1+c2+2), Mod 13:
        // Original values: [1,2,3,4,5]
        // Hashed leaves (HL): [H(1)=3, H(2)=5, H(3)=7, H(4)=9, H(5)=11]
        // Padded HL (to 8): [3,5,7,9,11,11,11,11] -> L0 to L7
        // Target leaf: L1 = H(2) = 5.
        // Path for L1 (sibling of L1 is L0=3):
        //   1. Sibling of L1 is L0=3.
        //   Parent P_L0_0 = HP(L0,L1) = HP(3,5) = 3+5+2=10.
        //   2. Sibling of P_L0_0 is P_L0_1 = HP(L2,L3) = HP(7,9) = 7+9+2=18 (5 mod 13).
        //   Parent P_L1_0 = HP(P_L0_0, P_L0_1) = HP(10,5) = 10+5+2=17 (4 mod 13).
        //   3. Sibling of P_L1_0 is P_L1_1.
        //      P_L0_2 = HP(L4,L5) = HP(11,11) = 11+11+2=24 (11 mod 13).
        //      P_L0_3 = HP(L6,L7) = HP(11,11) = 11+11+2=24 (11 mod 13).
        //      P_L1_1 = HP(P_L0_2, P_L0_3) = HP(11,11) = 11+11+2=24 (11 mod 13).
        // Path: [L0, P_L0_1, P_L1_1] = [FE(3), FE(5), FE(11)]
        assert_merkle_path(&proof.merkle_path, &[FE::new(3), FE::new(5), FE::new(11)]);
        assert!(proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, 1, &FE::new(2)));
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn merkle_proof_verifies_after_serialization_and_deserialization() {
        let values: Vec<Ecgfp5FE> = (1..6).map(Ecgfp5FE::new).collect();
        let merkle_tree = TestMerkleTreeEcgfp::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(1).unwrap();
        let serialize_proof = proof.serialize();
        let proof: TestProofEcgfp5 = Proof::deserialize(&serialize_proof).unwrap();
        assert!(proof.verify::<TestBackend<Ecgfp5>>(&merkle_tree.root, 1, &Ecgfp5FE::new(2)));
    }

    #[test]
    fn create_a_merkle_tree_with_10000_elements_and_verify_that_an_element_is_part_of_it() {
        let values: Vec<Ecgfp5FE> = (1..10000).map(Ecgfp5FE::new).collect();
        let merkle_tree = TestMerkleTreeEcgfp::build(&values).unwrap();
        let proof = merkle_tree.get_proof_by_pos(9349).unwrap();
        assert!(proof.verify::<TestBackend<Ecgfp5>>(&merkle_tree.root, 9349, &Ecgfp5FE::new(9350)));
    }

    fn assert_merkle_path(values: &[FE], expected_values: &[FE]) {
        for (node, expected_node) in values.iter().zip(expected_values) {
            assert_eq!(node, expected_node);
        }
    }

    #[test]
    fn verify_merkle_proof_for_single_value() {
        const MODULUS: u64 = 13;
        type U64PF = U64PrimeField<MODULUS>;
        type FE = FieldElement<U64PF>;

        let values: Vec<FE> = vec![FE::new(1)]; // Single element
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();

        // Update the expected root value based on the actual logic of TestBackend
        // New H_L(1) = 1+1+1 = 3. Root is FE(3).
        let expected_root = FE::new(3);
        assert_eq!(
            merkle_tree.root, expected_root,
            "The root of the Merkle tree does not match the expected value."
        );

        // Verify the proof for the single element
        let proof = merkle_tree.get_proof_by_pos(0).unwrap();
        assert!(
            proof.verify::<TestBackend<U64PF>>(&merkle_tree.root, 0, &values[0]),
            "The proof verification failed for the element at position 0."
        );
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn serialize_deserialize_empty_proof() {
        let empty_proof = TestProofEcgfp5 { merkle_path: Vec::new() };
        let serialized = empty_proof.serialize();
        assert!(serialized.is_empty(), "Serialized empty proof should be empty.");

        let deserialized: TestProofEcgfp5 = Proof::deserialize(&serialized).unwrap();
        assert!(deserialized.merkle_path.is_empty(), "Deserialized proof should have an empty merkle_path.");
    }

    // Helper struct for testing variable length serialization
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct VariableLengthNode {
        data: Vec<u8>,
    }

    impl VariableLengthNode {
        fn new(data: Vec<u8>) -> Self {
            Self { data }
        }
    }

    #[cfg(feature = "alloc")]
    impl Serializable for VariableLengthNode {
        fn serialize(&self) -> Vec<u8> {
            // The Proof serialization logic adds its own length prefix for the whole node.
            // This method just returns the raw variable data of the node.
            self.data.clone()
        }
    }

    #[cfg(feature = "alloc")]
    impl Deserializable for VariableLengthNode {
        fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError> {
            // This node type consumes all bytes passed to it as its data.
            Ok(VariableLengthNode { data: bytes.to_vec() })
        }
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn serialize_deserialize_proof_with_variable_length_nodes() {
        type ProofVariable = Proof<VariableLengthNode>;

        let nodes = vec![
            VariableLengthNode::new(vec![1, 2, 3]),        // 3 bytes
            VariableLengthNode::new(vec![4, 5, 6, 7, 8]), // 5 bytes
            VariableLengthNode::new(vec![9]),              // 1 byte
        ];
        let original_proof = ProofVariable { merkle_path: nodes };

        let serialized = original_proof.serialize();

        // Expected serialization:
        // len1 (4 bytes) | data1 (3 bytes) | len2 (4 bytes) | data2 (5 bytes) | len3 (4 bytes) | data3 (1 byte)
        // 0,0,0,3        | 1,2,3           | 0,0,0,5        | 4,5,6,7,8       | 0,0,0,1        | 9
        let expected_serialization: Vec<u8> = vec![
            0,0,0,3,  1,2,3,
            0,0,0,5,  4,5,6,7,8,
            0,0,0,1,  9,
        ];
        assert_eq!(serialized, expected_serialization, "Serialized output does not match expected format.");

        let deserialized: ProofVariable = Proof::deserialize(&serialized).unwrap();

        assert_eq!(original_proof.merkle_path.len(), deserialized.merkle_path.len(), "Merkle path lengths differ.");
        for (original_node, deserialized_node) in original_proof.merkle_path.iter().zip(deserialized.merkle_path.iter()) {
            assert_eq!(original_node, deserialized_node, "Node data differs after deserialization.");
        }
    }
}
