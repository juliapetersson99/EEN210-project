from fhirclient.models.medicationstatement import MedicationStatement
from fhirclient.models.condition import Condition
from fhirclient.models.patient import Patient


def fetch_medications(smart, patient):
    search = MedicationStatement.where(struct={"subject": patient.id})

    bundle = search.perform(smart.server)

    if not bundle.entry:
        return []

    return [entry.resource for entry in bundle.entry]


def fetch_conditions(smart, patient):
    search = Condition.where(struct={"subject": patient.id})

    bundle = search.perform(smart.server)

    if not bundle.entry:
        return []

    return [entry.resource for entry in bundle.entry]


def demo_patient():
    data = {
        "id": "130753",
        "active": True,
        "birthDate": "1979-01-01",
        "gender": "male",
        "name": [{"family": "Doe", "given": ["John "], "text": "John Doe"}],
        "resourceType": "Patient",
    }
    patient = Patient(data)
    patient.medications = []
    patient.conditions = []
    return data
