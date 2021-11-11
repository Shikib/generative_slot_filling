import json

def load_fn(fn):
    data = json.load(open(fn))
    examples = []
    for example in data:
        text = example['userInput']['text']
        slots = {}

        for label in example.get('labels', []):
            key = label['slot']
            value = text[label['valueSpan'].get('startIndex', 0):label['valueSpan']['endIndex']]
            slots[key] = value

        if 'context' in example and 'requestedSlots' in example['context']:
            requested_slots = example['context']['requestedSlots']
        else:
            requested_slots = []

        examples.append((requested_slots, text, slots))

    return examples
