# WhatsApp Messaging API

A simple Python function to send WhatsApp messages using the Twilio API.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Sign up for a Twilio account at [twilio.com](https://www.twilio.com) if you don't have one already.

3. Set up your environment variables:
   ```
   export TWILIO_ACCOUNT_SID="your_account_sid"
   export TWILIO_AUTH_TOKEN="your_auth_token"
   export TWILIO_WHATSAPP_NUMBER="your_twilio_whatsapp_number"
   ```

   Or create a `.env` file in the project root with these variables.

4. Configure WhatsApp in your Twilio account:
   - Go to the Twilio Console
   - Navigate to Messaging > Try it out > Send a WhatsApp message
   - Follow the instructions to set up your WhatsApp sandbox

## Usage

```python
from src.whatsapp_api.main import send_whatsapp_message

# Send a message to a specific number
recipient = "+1234567890"  # Include country code
message = "Hello from your WhatsApp API!"
success = send_whatsapp_message(recipient, message)

if success:
    print("Message sent successfully!")
else:
    print("Failed to send message.")
```

## Notes

- The recipient must be opted-in to receive messages from your Twilio WhatsApp number.
- In sandbox mode, you can only send messages to verified numbers.
- For production use, you need to apply for a WhatsApp Business Profile through Twilio. 