import os
import time
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

def send_whatsapp_message(to_number, message):
    """
    Send a WhatsApp message to a specific number using Twilio.
    
    Args:
        to_number (str): The recipient's phone number with country code (e.g., '+1234567890')
        message (str): The message to send
        
    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    # Your Twilio Account SID and Auth Token
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    
    # Your Twilio WhatsApp number (must be configured in Twilio)
    from_number = os.environ.get('TWILIO_WHATSAPP_NUMBER')
    
    # Debug information
    print(f"Account SID: {account_sid}")
    print(f"From number: {from_number}")
    print(f"To number: {to_number}")
    
    if not all([account_sid, auth_token, from_number]):
        print("Error: Missing Twilio credentials. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_NUMBER environment variables.")
        return False
    
    try:
        # Initialize Twilio client
        client = Client(account_sid, auth_token)
        
        # Format the WhatsApp number if needed
        if not to_number.startswith('whatsapp:'):
            to_number = f"whatsapp:{to_number}"
        
        # Format the from number correctly
        if not from_number.startswith('whatsapp:'):
            from_number = f"whatsapp:{from_number}"
        
        print(f"Sending message from {from_number} to {to_number}")
        
        # Send the message with the actual message content
        message_obj = client.messages.create(
            from_=from_number,
            body=message,  # Use the actual message instead of the test message
            to=to_number
        )
        
        print(f"Message sent successfully! SID: {message_obj.sid}")
        
        # Check message status
        print("Checking message status...")
        time.sleep(2)  # Wait a moment for the message to be processed
        
        message_status = client.messages(message_obj.sid).fetch()
        print(f"Message status: {message_status.status}")
        
        # Print all available attributes of the message for debugging
        print("\nMessage details:")
        for attr in dir(message_status):
            if not attr.startswith('_') and not callable(getattr(message_status, attr)):
                try:
                    value = getattr(message_status, attr)
                    print(f"{attr}: {value}")
                except Exception as e:
                    print(f"Could not get {attr}: {str(e)}")
        
        if message_status.status in ['delivered', 'sent', 'queued']:
            print("Message was queued, sent, or delivered successfully.")
        elif message_status.status == 'failed':
            print(f"Message failed to send. Error: {message_status.error_message}")
            print(f"Error code: {message_status.error_code}")
        elif message_status.status == 'undelivered':
            print("Message was not delivered. The recipient may not be opted in.")
        else:
            print(f"Message status is {message_status.status}. This may indicate the message is still processing.")
        
        return True
    
    except Exception as e:
        print(f"Error sending message: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    # Example phone number (with country code)
    recipient = "+558192249327"
    message_text = "Testing"  # Using the same message that worked in the console
    
    send_whatsapp_message(recipient, message_text)
