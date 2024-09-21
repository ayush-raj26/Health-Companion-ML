import csv
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to generate data with increasing blood pressure and blood sugar levels
def generate_increasing_data(name):
    data = []
    blood_pressure = random.uniform(120, 130)
    blood_sugar = random.uniform(70, 110)
    for i in range(1, 26):
        date = f'2023-{int(i/2) + 1:02d}-{15*(i%2) + 1:02d}'
        blood_pressure += random.uniform(0.7, 1.5)
        blood_sugar += random.uniform(1, 3)
        data.append([name, age, height, weight, gender, round(blood_sugar, 2), round(blood_pressure, 2), date, True,True ])
    return data

# Generate CSV data for the second dataset
with open('patient_data.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # writer.writerow(['Name', 'Age', 'Height', 'Weight', 'Gender', 'Blood Sugar Level', 'Blood Pressure', 'Date of Visit', 'Is Heart Patient', 'Is Sugar Patient'])
    
    names = [f'Patient_{i}' for i in range(131, 136)]
    
    for name in names:
        age = random.randint(20, 80)
        height = random.randint(150, 200)
        weight = random.randint(50, 100)
        gender = random.choice(['Male', 'Female'])
        
        data = generate_increasing_data(name)
        writer.writerows(data)
