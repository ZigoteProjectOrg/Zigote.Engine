/// Lightweight entity identifier. Zero is reserved as "null entity".
pub const Entity = u32;
pub const null_entity: Entity = 0;

pub const EntityGen = struct {
    next: Entity = 1,

    pub fn create(self: *EntityGen) Entity {
        const id = self.next;
        self.next += 1;
        return id;
    }
};
