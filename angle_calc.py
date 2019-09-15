import numpy
import math

def angle(orient1, orient2):
	angle_1=abs(orient1-orient2)
	angle_2=numpy.pi-(abs(orient1-orient2))
	if angle_1<angle_2:
		return angle_1
	else:
		return angle_2

def AngDiff(orient1, orient2):
	Ang = math.atan2(math.sin(orient1 - orient2), math.cos(orient1 - orient2))
	return Ang
			
if __name__ == '__main__':
	A=-170 #First angle
	B=170 #Second angle
	print("Start of angle calculation")
	anglea=numpy.rad2deg(AngDiff(numpy.deg2rad(A),numpy.deg2rad(B)))
	print("Angle between the two orientations is",anglea)
	print("Angle calculation completed")