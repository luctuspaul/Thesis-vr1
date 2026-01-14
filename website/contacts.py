import csv
import os

CONTACTS_FILE = "website/contacts.csv"

def save_contact(name, email, phone):
    file_exists = os.path.exists(CONTACTS_FILE)

    with open(CONTACTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Name", "Email", "Phone"])

        writer.writerow([name, email, phone])
