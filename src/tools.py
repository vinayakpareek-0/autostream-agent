"""Lead capture tool — mock API for collecting qualified lead data."""


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Simulate a CRM API call to capture a qualified lead.

    In production, this would POST to a CRM like HubSpot or Salesforce.
    """
    print(f"[SUCCESS] Lead captured: {name}, {email}, {platform}")
    return f"Lead captured successfully: {name}, {email}, {platform}"
