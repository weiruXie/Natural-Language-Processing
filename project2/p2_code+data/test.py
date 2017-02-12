with open("transparser.py", "rb") as f:
	data = f.read()

with open("transparser_new.py", "wb") as f:
	f.write(data.replace("    ", "\t"))