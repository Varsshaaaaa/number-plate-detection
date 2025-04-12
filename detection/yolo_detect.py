import json

def load_stolen_plates(path='data/stolen_plates.json'):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return {}

def check_plate_alert(plate_text, db):
    plate_text = plate_text.replace(' ', '').upper()
    return db.get(plate_text, None)
