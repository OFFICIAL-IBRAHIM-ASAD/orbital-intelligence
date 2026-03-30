from pydantic import BaseModel
from typing import List, Optional

# Define the data structure for a military or corporate claim
class Claim(BaseModel):
    id: int
    source: str
    claim_type: str # e.g., "Kinetic Strike", "Policy Restriction"
    description: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    status: str = "Pending Verification"

class LedgerDatabase:
    def __init__(self):
        # A mock database seeded with thesis-relevant examples
        self.claims = [
            Claim(
                id=1,
                source="OSINT Report (Feb 2026)",
                claim_type="Kinetic Strike",
                description="Algorithmic targeting utilized to neutralize logistics hub near Bandar Abbas.",
                latitude=27.15, 
                longitude=56.25,
                status="Pending Verification"
            ),
            Claim(
                id=2,
                source="Anthropic AUP",
                claim_type="Policy Restriction",
                description="Strict prohibition on military targeting or weaponized API integrations.",
                status="Verified"
            )
        ]

    def get_all_claims(self) -> List[Claim]:
        return self.claims

    def get_claim_by_id(self, claim_id: int) -> Claim:
        for claim in self.claims:
            if claim.id == claim_id:
                return claim
        return None

# Instantiate the database
db = LedgerDatabase()
