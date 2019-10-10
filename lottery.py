# Import Pandas
import pandas as pd

# Read lottery csv file and store in dataframe
data = pd.read_csv('lottery.csv')

# Store slot names
lottery_slots = ['first','second','third','fourth','fifth','sixth','bonus']

# Initialize frequency Dataframe
frequency_df =  pd.DataFrame(columns=['count'],index=range(1,46))
frequency_df = frequency_df.fillna(0)

# Tabluate count
for slot in lottery_slots:
    slot_data = data[slot].value_counts()
    for key in slot_data.keys():
        frequency_df['count'][key] += slot_data[key]

print(frequency_df)

frequency_df = frequency_df.sort_values('count',ascending=False)
print(frequency_df)

for k,v in frequency_df.iterrows():
    n = v['count']
    print(f"{k} -> {n}")

# print(frequency_d)
# n = sum(frequency_d.values())
# print(n==7*716)
# frequency_d[1] = data['first'].value_counts()
# frequency_d[2] = data['second'].value_counts()
# frequency_d[3] = data['third'].value_counts()
# frequency_d[4] = data['fourth'].value_counts()
# frequency_d[5] = data['fifth'].value_counts()
# frequency_d[6] = data['sixth'].value_counts()
# frequency_d[7] = data['bonus'].value_counts()

# print(data[['first','second']].value_counts())
# print(data[['first','second']].keys())
# print(data['first'].value_counts().keys())
# print(frequency_d)
# print(type(frequency_d))

