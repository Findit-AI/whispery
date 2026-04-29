//! `ChunkId` — monotonic identifier for emitted transcript chunks.

/// Monotonic identity for a chunk within a single `Transcriber`
/// lifetime. Increases by 1 per emitted chunk (including chunks
/// that produce `Event::Error`); suitable as a lancedb primary
/// key. See spec §4.2 and §5.5.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChunkId(u64);

impl ChunkId {
    /// Construct from a raw `u64`. Crate-private; the dispatch state
    /// machine is the only legitimate constructor.
    pub(crate) const fn from_raw(n: u64) -> Self {
        Self(n)
    }

    /// Raw underlying value.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl core::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_through_raw() {
        let c = ChunkId::from_raw(42);
        assert_eq!(c.as_u64(), 42);
    }

    #[test]
    fn ordering_is_total_and_numeric() {
        let a = ChunkId::from_raw(1);
        let b = ChunkId::from_raw(2);
        assert!(a < b);
        assert_eq!(a, ChunkId::from_raw(1));
    }
}
