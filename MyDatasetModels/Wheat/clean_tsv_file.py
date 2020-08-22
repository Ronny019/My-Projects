file1 = open("./seeds_dataset.tsv","r")
file2 = open("./cleaned_seeds_dataset.csv","w")

string_list = file1.readlines()

for entry in string_list:
    entry_list = entry.split("\t")
    counter = 0
    for token in entry_list:
        file2.write(token)
        counter = counter + 1
        if counter != len(entry_list) and token != '':
            file2.write(",")

file2.close()
file1.close()