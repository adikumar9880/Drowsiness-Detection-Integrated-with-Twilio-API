# Configuration Example for Drowsiness Detection System
# Copy this file to config.py and fill in your actual credentials
# Never commit config.py to version control!

# Twilio Configuration
TWILIO_ACCOUNT_SID = 'your_account_sid_here'
TWILIO_AUTH_TOKEN = 'your_auth_token_here'
TWILIO_PHONE_NUMBER = 'your_twilio_phone_number_here'
EMERGENCY_CONTACTS = ['+919447224478', '+1234567890']

# Example usage in your main script:
# from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, EMERGENCY_CONTACTS
