print("20000 17 26")
with open("examples3_unprocess.data", "r") as f:
    for line in f:
        sp_line = line.strip().split(",")
        place = (ord(sp_line[0]) - 65)
        print(place, " ".join(sp_line[1:]), " ".join(str(int(i == place)) for i in range(26)))