use alloc::vec::Vec;
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// A backend for Merkle trees. This defines raw `Data` from which the Merkle
/// tree is built from. It also defines the `Node` type and the hash functions
/// used to build parent nodes from children nodes.
///
/// Implementers of this trait should pay careful attention to **domain separation**
/// in their hashing logic to ensure the security of the Merkle tree. This means
/// that hashes for leaves and internal nodes should be computed differently, even if
/// the underlying data might coincidentally be the same. This prevents an attacker
/// from crafting a leaf that could be misinterpreted as an internal node, or vice-versa.
/// See the documentation for `hash_data` and `hash_new_parent` for specific guidance.
pub trait IsMerkleTreeBackend {
    type Node: PartialEq + Eq + Clone + Sync + Send;
    type Data: Sync + Send;

    /// This function takes a single variable `Data` and converts it to a `Node` (a leaf hash).
    ///
    /// **Domain Separation for Leaf Nodes:**
    /// To prevent type confusion attacks (e.g., an attacker crafting leaf data that hashes
    /// to the same value as a pre-existing internal node), implementations **must** ensure
    /// that the hashing process for leaf data is distinct from the hashing process for
    /// internal nodes (as performed by `hash_new_parent`).
    ///
    /// A common way to achieve this is by prepending a domain-specific constant prefix
    /// to the data before hashing. For example:
    /// `hash_function(LEAF_DOMAIN_TAG || serialize(leaf_data))`
    /// where `LEAF_DOMAIN_TAG` is a constant unique to leaf hashing (e.g., `0x00`).
    fn hash_data(leaf: &Self::Data) -> Self::Node;

    /// This function takes the list of data from which the Merkle
    /// tree will be built from and converts it to a list of leaf nodes.
    fn hash_leaves(unhashed_leaves: &[Self::Data]) -> Vec<Self::Node> {
        #[cfg(feature = "parallel")]
        let iter = unhashed_leaves.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = unhashed_leaves.iter();

        iter.map(|leaf| Self::hash_data(leaf)).collect()
    }

    /// This function takes two children nodes (`child_1`, `child_2`) and builds a new parent `Node`.
    /// It will be used in the construction of the Merkle tree.
    ///
    /// **Domain Separation for Internal Nodes:**
    /// To prevent type confusion attacks, implementations **must** ensure that the hashing
    /// process for internal nodes is distinct from the hashing process for leaf data
    /// (as performed by `hash_data`).
    ///
    /// A common way to achieve this is by prepending a domain-specific constant prefix
    /// to the concatenated data of the children nodes before hashing. For example:
    /// `hash_function(NODE_DOMAIN_TAG || serialize(child_1) || serialize(child_2))`
    /// where `NODE_DOMAIN_TAG` is a constant unique to internal node hashing (e.g., `0x01`),
    /// and distinct from any tag used in `hash_data`.
    fn hash_new_parent(child_1: &Self::Node, child_2: &Self::Node) -> Self::Node;
}
