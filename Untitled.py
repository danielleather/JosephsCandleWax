import cairo
import math
import os
import imageio
import shutil
from scipy.integrate import quad
import sys
import numpy as np
import xml.etree.ElementTree as ET

"""
filepath = os.getcwd() + '/file.svg'
tree = ET.parse(filepath)
root = tree.getroot()
path_element = root.find('path')
path = path_element.get('d')
"""

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


path_attribute = "M 261.10235595703125 79.82002258300781 L 128.00900268554688 184.22947692871094 L 197.9977569580078 185.3768310546875 L 133.74578857421875 251.9235076904297 L 195.70303344726562 250.77615356445312 L 133.74578857421875 320.764892578125 L 196.8503875732422 318.4701843261719 L 130.3037109375 390.753662109375 L 234.71316528320312 388.4589538574219 L 235.8605194091797 512.3734741210938 L 290.9336242675781 511.2261047363281 L 286.3442077636719 387.31158447265625 L 375.8380126953125 387.31158447265625 L 321.9122619628906 320.764892578125 L 375.8380126953125 319.6175537109375 L 319.6175537109375 254.21823120117188 L 374.690673828125 253.07086181640625 L 313.8807678222656 185.3768310546875 L 367.8065185546875 185.3768310546875 L 274.8706359863281 84.4094467163086"

iterations = 60

line_equation = lambda x, m, b : m*x + b

# functionscape = [[((x_path, (m, b)), (lower_bound,upper_bound))], [((y_path, (m, b)), (lower_bound,upper_bound))]]

#path = [[0.7,0.9],[0.2,0.9],[0.2,0.5],[0.7,0.5],[0.7,0.1],[0.2,0.1],[0.701,0.901]]
unscaled_path = []
path = []

path_split = path_attribute.split(" ")

j = 0
y_coordinates = []
x_coordinates = []

for i in path_split:
    if not is_number(i):
        continue


    if j % 2 == 0:
        x_coordinates.append(float(i))
        j += 1
        continue

    y_coordinates.append(float(i))
    j += 1


highest_y_coordinate = max(y_coordinates)
highest_x_coordinate = max(x_coordinates)

graph_y_range = highest_y_coordinate * 1.2
graph_x_range = highest_x_coordinate * 1.2

x_coordinate = 0
j = 1

for i in path_split:
    if not is_number(i):
        continue

    if j % 2 == 0:
        unscaled_path.append([x_coordinate, graph_y_range - float(i)])
        j += 1
        continue

    x_coordinate = float(i)
    j += 1

for i in unscaled_path:
    path.append([i[0]/graph_x_range, i[1]/graph_y_range])

total_path_length = 0
previous_point = []
for i in path:
    if path.index(i) == 0:
        previous_point = i
        continue

    x_diff = i[0] - previous_point[0]
    y_diff = i[1] - previous_point[1]


    path_length = math.sqrt((x_diff)**2 + (y_diff)**2)
    total_path_length += path_length

    previous_point = i


functionscape = [[],[]]
previous_point = path[0]
lower_bound = 0
for i in path:
    if path.index(i) == 0:
        continue

    distance_from_last_point = math.sqrt((i[0] - previous_point[0])**2 + (i[1] - previous_point[1])**2)/total_path_length
    upper_bound = lower_bound + distance_from_last_point

    m_x = (i[0] - previous_point[0])/(upper_bound - lower_bound)
    b_x = i[0] - m_x*upper_bound
    functionscape[0].append(((line_equation, (m_x, b_x)), (lower_bound, upper_bound)))

    m_y = (i[1] - previous_point[1])/(upper_bound - lower_bound)
    b_y = i[1] - m_y*upper_bound
    functionscape[1].append(((line_equation, (m_y, b_y)), (lower_bound, upper_bound)))

    lower_bound = upper_bound
    previous_point = i


series = 'cos'

plotrange = 1

def f_by_cos(x,n,L,f,m,b):
    return f(x,m,b)*math.cos(n*math.pi*x/L)

def f_by_sin(x,n,L,f,m,b):
    return f(x,m,b)*math.sin(n*math.pi*x/L)

def half_range_sin_coefficient(L, n, functionscape):
    result = 0
    for i in functionscape:
        m = i[0][1][0]
        b = i[0][1][1]
        anon_func = i[0][0]
        result += quad(f_by_sin, i[1][0], i[1][1], args=(n,L,anon_func,m,b))[0]
    return 2*result/L

def first_half_range_cos_coefficient(L, functionscape):
    result = 0
    for i in functionscape:
        m = i[0][1][0]
        b = i[0][1][1]
        anon_func = i[0][0]
        result += quad(anon_func, i[1][0], i[1][1], args=(m,b))[0]
    return result/L

def consequent_half_range_cos_coefficient(L, n, functionscape):
    result = 0
    for i in functionscape:
        m = i[0][1][0]
        b = i[0][1][1]
        anon_func = i[0][0]
        result += quad(f_by_cos, i[1][0], i[1][1], args=(n,L,anon_func,m,b))[0]
    return 2*result/L



if (series == 'cos'):
    circle_dimensions = [(first_half_range_cos_coefficient(plotrange, functionscape[0]), 0, math.pi/2), (first_half_range_cos_coefficient(plotrange, functionscape[1]), 0, 0)]
elif (series == 'sin'):
    circle_dimensions = []

for n in range(1,iterations):
    if (series == 'cos'):
        coefficient = consequent_half_range_cos_coefficient(plotrange,n,functionscape[0])
        if (coefficient > 0.0001):
            circle_dimensions.append((500*coefficient, n/2, math.pi/2))
            circle_dimensions.append((500*coefficient, -n/2, math.pi/2))
        elif (coefficient < -0.0001):
            circle_dimensions.append((-500*coefficient, n/2, 3*math.pi/2))
            circle_dimensions.append((-500*coefficient, -n/2, 3*math.pi/2))

        coefficient = consequent_half_range_cos_coefficient(plotrange,n,functionscape[1])
        if (coefficient > 0.0001):
            circle_dimensions.append((500*coefficient, n/2, 0))
            circle_dimensions.append((500*coefficient, -n/2, 0))
        elif (coefficient < -0.0001):
            circle_dimensions.append((-500*coefficient, n/2, math.pi))
            circle_dimensions.append((-500*coefficient, -n/2, math.pi))

    elif (series == 'sin'):
        coefficient = half_range_sin_coefficient(plotrange,n,functionscape[0])
        if (coefficient > 0.0001):
            circle_dimensions.append((500*coefficient, n/2, 0))
            circle_dimensions.append((500*coefficient, -n/2, 0))
        elif (coefficient < -0.0001):
            circle_dimensions.append((-500*coefficient, n/2, math.pi))
            circle_dimensions.append((-500*coefficient, -n/2, math.pi))

        coefficient = half_range_sin_coefficient(plotrange,n,functionscape[1])
        if (coefficient > 0.0001):
            circle_dimensions.append((500*coefficient, n/2, math.pi/2))
            circle_dimensions.append((500*coefficient, -n/2, math.pi/2))
        elif (coefficient < -0.0001):
            circle_dimensions.append((-500*coefficient, n/2, 3*math.pi/2))
            circle_dimensions.append((-500*coefficient, -n/2, 3*math.pi/2))


for i in circle_dimensions:
    print(i)



# raise SystemExit

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


FRAMECOUNT = 300

centre = [500,1600]

frame_dimensions = (2100,2100)

'''
circle_dimension[0] = radius of circle
circle_dimension[1] = number of times circle rotates around the origin.
'''

if (os.path.isdir(os.getcwd() + '/workdir')):
    print('~removing previous workdir~')
    shutil.rmtree(os.getcwd() + '/workdir/')

#find the sum of all x values (indice = 0) or y values (indice = 1) excluding the last one
def pointsSum(points, indice):
    sum = 0

    for i in points[:-1]:
        sum += i[indice]

    return sum

def fill_past_points(ctx, points_collection):
    ctx.set_source_rgb(1,1,1)
    ctx.set_line_width(5)
    ctx.move_to(points_collection[0][0], points_collection[0][1])
    for i in range(len(points_collection) - 1):
        ctx.arc(points_collection[i][0], points_collection[i][1], 2, 0, math.pi*2)
        ctx.fill()

        if (i == 0):
            continue

        ctx.move_to(points_collection[i - 1][0],points_collection[i - 1][1])
        ctx.line_to(points_collection[i][0],points_collection[i][1])
        ctx.stroke()
    ctx.set_line_width(2)


points_collection = []

filenames = []
cwd = os.getcwd()

os.mkdir(cwd + '/workdir')

t = 0
onedxplot = []
onedyplot = []
centre_point = centre

for x in range(FRAMECOUNT):
    #create image surface and context
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,frame_dimensions[0],frame_dimensions[1])
    ctx = cairo.Context (surface)

    #black background
    ctx.set_source_rgb(0,0,0)
    ctx.rectangle(0,0,frame_dimensions[0],frame_dimensions[1])
    ctx.fill()

    #create center point
    ctx.set_source_rgb(1,1,1)
    ctx.arc(centre_point[0],centre_point[1],10,0,2*math.pi)
    ctx.fill()

    points = [centre_point]



    for i in circle_dimensions:

#         find angle offset
        if (series == 'sin'):
            offset = math.pi
        elif (series == 'cos'):
            offset = 3*math.pi/2

        theta = offset + (x/FRAMECOUNT)*2*math.pi*i[1] + i[2]

        xphase = i[0]*math.cos(theta)
        yphase = i[0]*math.sin(theta)
        points.append((xphase, yphase))

        finalx = pointsSum(points, 0) + xphase
        finaly = pointsSum(points, 1) + yphase

        #colour in point
        ctx.set_source_rgb(1,1,1)
        ctx.arc(finalx,finaly,5,0,2*math.pi)
        ctx.fill()

        ctx.arc(pointsSum(points, 0), pointsSum(points, 1), i[0], 0, 2*math.pi)
        ctx.stroke()

        #if we are on the last circle, fill in the past points and add the current one to the collection
        if i == circle_dimensions[-1]:
            points_collection.append((finalx, finaly))
            fill_past_points(ctx, points_collection)

    radii_position = [500,1600]

    for i in points[1:]:
        ctx.move_to(radii_position[0], radii_position[1])
        radii_position[0] = radii_position[0] + i[0]
        radii_position[1] = radii_position[1] + i[1]
        ctx.line_to(radii_position[0], radii_position[1])

    ctx.stroke()

    #horizontal and vertical tracking
    onedyplot.append((1100 + t, points_collection[-1][1]))
    ctx.move_to(points_collection[-1][0], points_collection[-1][1])
    ctx.line_to(1100 + t, points_collection[-1][1])
    ctx.stroke()

    onedxplot.append((points_collection[-1][0], 1100 - t))
    ctx.move_to(points_collection[-1][0], points_collection[-1][1])
    ctx.line_to(points_collection[-1][0], 1100 - t)
    ctx.stroke()

    if (len(onedyplot) > 1):
        for k in range(len(onedyplot) - 1):
            ctx.move_to(onedyplot[k][0], onedyplot[k][1])
            ctx.line_to(onedyplot[k+1][0], onedyplot[k+1][1])
            ctx.stroke()

            ctx.move_to(onedxplot[k][0], onedxplot[k][1])
            ctx.line_to(onedxplot[k+1][0], onedxplot[k+1][1])
            ctx.stroke()



    surface.write_to_png('workdir/Frame' + str(x) + '.png')

    filenames.append(cwd + '/workdir/Frame' + str(x) + '.png')

    if (x % 10 == 0):
        print('doing frame ' + str(x))


    t += 1000/FRAMECOUNT

#create gif
images = []
for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave(cwd + '/final.gif', images)

#remove images after gif is created
shutil.rmtree(cwd + '/workdir/')

print('finished')
