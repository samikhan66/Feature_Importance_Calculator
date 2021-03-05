def string2bool(string):
	"""Converts string to boolean"""
	if string == "True":
		return True
	elif string == "False":
		return False
	elif string == "None":
		return None
	else:
		return string