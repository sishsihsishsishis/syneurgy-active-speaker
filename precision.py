import csv
import matplotlib.pyplot as plt

def parse_csv(filepath):
    data = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            speaker_id = row[0].strip()
            start_time = float(row[1].strip())
            end_time = float(row[2].strip())
            x1 = int(row[3].strip())
            y1 = int(row[4].strip())
            x2 = int(row[5].strip())
            y2 = int(row[6].strip())
            data.append((speaker_id, start_time, end_time, x1, y1, x2, y2))
    return data

def calculate_precision(predictions, bbox_ranges):
    precision_results = {}
    incorrect_bbox = []

    for pred in predictions:
        speaker_id, start_time, end_time, x1, y1, x2, y2 = pred
        bbox_range = bbox_ranges.get(speaker_id, {})
        
        correct = False
        for range_key, (range_min, range_max) in bbox_range.items():
            if range_min <= x1 <= range_max:
                correct = True
                break
        
        if speaker_id not in precision_results:
            precision_results[speaker_id] = {'total': 0, 'correct': 0}
        
        precision_results[speaker_id]['total'] += 1
        if correct:
            precision_results[speaker_id]['correct'] += 1
        else:
            incorrect_bbox.append((speaker_id, start_time, end_time, x1, y1, x2, y2))

    for speaker_id, results in precision_results.items():
        precision = (results['correct'] / results['total']) * 100 if results['total'] > 0 else 0
        precision_results[speaker_id]['precision'] = precision
    
    return precision_results

def plot_histogram(predictions, speaker_id, bbox_ranges, precision_results):
    x1_values = [x1 for (spk, _, _, x1, _, _, _) in predictions if spk == speaker_id]
    bbox_range = bbox_ranges.get(speaker_id, {})
    precision = precision_results[speaker_id]['precision']
    total_samples = precision_results[speaker_id]['total']
    correct_samples = precision_results[speaker_id]['correct']
    
    plt.hist(x1_values, bins=30, alpha=0.5, label=f'Total number of entries: {len(x1_values)}')
    
    for range_key, (range_min, range_max) in bbox_range.items():
        plt.axvline(range_min, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(range_max, color='r', linestyle='dashed', linewidth=1)
        num_correct_in_range = sum(range_min <= x1 <= range_max for x1 in x1_values)
        num_incorrect_in_range = sum(x1 < range_min or x1 > range_max for x1 in x1_values)
        plt.text((range_min + range_max) / 2, plt.ylim()[1] * 0.8, f'Correct: {num_correct_in_range}', color='r')
        plt.text((range_min + range_max) / 2, plt.ylim()[1] * 0.7, f'Incorrect: {num_incorrect_in_range}', color='b')
    
    plt.xlabel('X1')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of X1 for {speaker_id}\nPrecision: {precision:.2f}% | Total samples: {total_samples} | Correct samples: {correct_samples}')
    plt.legend()
    plt.show()
    plt.close()

def main():
    predictions_file = 'exports/active_speakers.csv'  # Replace with the path to your predictions CSV file

    predictions = parse_csv(predictions_file)
    
    bbox_ranges = {
        'speaker00': {
            'range_1': (200, 350)  # Left side of the screen
        },
        'speaker01': {
            'range_1': (700, 900)  # Right side of the screen
        }
    }
    
    precision_results = calculate_precision(predictions, bbox_ranges)
    for speaker_id, results in precision_results.items():
        print(f"{speaker_id} precision: {results['precision']:.2f}%")
        plot_histogram(predictions, speaker_id, bbox_ranges, precision_results)

if __name__ == "__main__":
    main()
