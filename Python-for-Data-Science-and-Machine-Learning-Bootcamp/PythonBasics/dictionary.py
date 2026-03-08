# Step 1: Create two dictionaries
teamA = {
    "member1": {"name": "Rushikesh", "age": 29, "city": "Hyderabad"},
    "member2": {"name": "Sneha", "age": 27, "city": "Chennai"},
    "member3": {"name": "Vikram", "age": 31, "city": "Delhi"}
}

teamB = {
    "member4": {"name": "Ravi", "age": 29, "city": "Hyderabad"},
    "member5": {"name": "Meera", "age": 27, "city": "Chennai"},
    "member6": {"name": "Ananya", "age": 31, "city": "Delhi"}
}

# # Step 2: Update values
# teamA["member1"].update({"name": "rushikesh"})   # change name
# teamA["member2"]["city"] = "Bengaluru"       # direct assignment

# # Step 3: Access values safely
# print(teamA["member1"]["name"].capitalize())    # Rushikesh
# print(teamA["member2"].get("city", "default"))  # Bengaluru

# # Step 4: Remove items
# removed = teamA.pop("member3")                  # remove member3
# print("Removed:", removed)

# last_removed = teamA.popitem()                  # remove last inserted
# print("Last removed:", last_removed)

# Step 5: Merge dictionaries
merged_dict = teamA | teamB                     # Python 3.9+ operator
# print("Merged with | :", merged_dict)

# merged = teamA.copy()
# merged.update(teamB)                            # update method
# print("Merged with update:", merged)

# Step 6: Iterate over dictionary
# for key, value in merged_dict.items():
#     print(f"{key} -> Name: {value['name']}, Age: {value['age']}, City: {value['city']}")

# # Step 7: Dictionary views
# print("Keys:", merged.keys())
# print("Values:", merged.values())
# print("Items:", merged.items())

# # Step 8: Clear dictionary
# merged.clear()
# print("After clear:", merged)

# Default values 
teamA.setdefault("member7", {"name": "Elon", "age": 56, "city": "NY city"})

# for key, value in teamA.items():
#     print(f"{key} => Name: {value['name']}, Age: {value['age']}, City: {value['city']}")
    
# Compressions
sqaures = {x:x**2 for x in range(11)}
# print(f"squares:{sqaures}")    

# Nested dictionaries 
companyA = {
    "teamA": {"employee1": {"name": "Andrej", "role": "deep learning engineeer"}},
    "teamB": {"employee2": {"name": "Sam", "role": 'AI engineer'}}
}

companyB = {
    "teamC": {"employee1": {"name": "Karpathy", "role": "deep learning engineeer"}},
    "teamD": {"employee2": {"name": "Alman", "role": 'AI engineer'}}
}

merger = companyA.copy()
merger.update(companyB)

# Iterate properly through nested dictionaries
# for team, employees in merger.items():
#     for emp_id, details in employees.items():
#         print(f"{team} - {emp_id} => Name: {details['name']}, Role: {details['role']}")
        
# deepcopy() (from copy module) → full copy including nested dicts.

sorted_by_key = dict(sorted(teamA.items()))
# print(sorted_by_key)

sorted_by_age = dict(sorted(teamA.items(), key=lambda x: x[1]["age"]))
# print(sorted_by_age)

# Membership Tests
# if {"name": "Sneha", "age": 27, "city": "Chennai"} in teamA.values(): print('Yes, member is present')
# else: print('No, member is not-present')
          
# Dictionary as counters 
from collections import Counter 
data = ["apple", "Banana", "Pine-Apple", "Watermelon", "Avacado", "Apple", "Banana"]
# print(Counter(data))

# Unpacking 
merged = {**teamA, **teamB}

# Advanced Methods
new_dict = dict.fromkeys(["a", "b", "c"], "vegeterians")
print(new_dict)

