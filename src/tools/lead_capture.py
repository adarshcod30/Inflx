def mock_lead_capture(name, email, platform):
    """
    Mock API function for lead capture.
    Only triggered when name, email, and platform are all collected.
    """
    # In a real scenario, this would be a POST request to a CRM or database.
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    return f"Lead Captured: {name} ({email}) on {platform}"
