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

path_attribute = input("Input svg path attribute here:")

iterations = 100
FRAMECOUNT = 250

#line_equation = lambda x, m, b : m*x + b
#quadratic_equation= lambda t, p1, p2, p3 : p1 * (1 - t) ** 2 + 2 * p2 * (1 - t) * t + p3 * t ** 2
#cubic_equation= lambda t, p1, p2, p3, p4 : p1 * (1 - t) ** 3 + 3 * p2 * t * (1 - t) ** 2 + 3 * p3 * t ** 2 * (1 - t) + p4 * t ** 3

#bezier equeations for linear, quadratic and cubic curves with a skew and offset applied to translate the curve
def linear_bezier(t, p0, p1, skew, offset):
    return p0*(1 - skew*(t - offset)) + p1*skew*(t - offset)

def quadratic_bezier(t, p0, p1, p2, skew, offset):
    return p0*(1 - skew*(t - offset))**2 + 2*p1*(1 - skew*(t - offset))*skew*(t - offset) + p2*(skew*(t - offset))**2

def cubic_bezier(t,p0,p1,p2,p3, skew, offset):
    return p0*(1 - skew*(t - offset))**3 + 3*p1*skew*(t - offset)*(1 - skew*(t - offset))**2 + 3*p2*(skew*(t - offset))**2*(1 - skew*(t - offset)) + p3*(skew*(t - offset))**3

unscaled_path = []
path = []

path_split = path_attribute.split(" ")

j = 0
y_coordinates = []
x_coordinates = []

#get a list of x coordinates and a list of y coordinates
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
#find the highest of each list and multiply by 1.2
graph_y_range = highest_y_coordinate # * 1.2
graph_x_range = highest_x_coordinate # * 1.2

x_coordinate = 0
j = 1

#flip y values to normal cartesion representation
for i in path_split:
    if not is_number(i):
        unscaled_path.append(i)
        continue

    if j % 2 == 0:
        unscaled_path.append([x_coordinate, graph_y_range - float(i)])
        j += 1
        continue

    x_coordinate = float(i)
    j += 1

#scale coordinates against size of graph, to fit in [0:1]
for i in unscaled_path:
    if not isinstance(i, list):
        path.append(i)
        continue

    path.append([i[0]/graph_x_range, i[1]/graph_y_range])

def find_path_length(base_point, end_point):
    x_diff = base_point[0] - end_point[0]
    y_diff = base_point[1] - end_point[1]
    return math.sqrt(x_diff ** 2 + y_diff ** 2)

def estimate_quadratic_length(base_point, anchor, end_point):
    segments = 2000
    domain = np.linspace(0, 1, segments)

    length= 0
    for n in range(1, segments):
        segment_x_start = quadratic_bezier(domain[n - 1], base_point[0], anchor[0], end_point[0], 1, 0)
        segment_y_start = quadratic_bezier(domain[n - 1], base_point[1], anchor[1], end_point[1], 1, 0)
        segment_start = [segment_x_start, segment_y_start]

        segment_x_end = quadratic_bezier(domain[n], base_point[0], anchor[0], end_point[0], 1, 0)
        segment_y_end = quadratic_bezier(domain[n], base_point[1], anchor[1], end_point[1], 1, 0)
        segment_end = [segment_x_end, segment_y_end]

        length += find_path_length(segment_start, segment_end)

    return length

def estimate_cubic_length(base_point, anchor1, anchor2, end_point):
    segments = 2000
    domain = np.linspace(0, 1, segments)

    length= 0
    for n in range(1, segments):
        segment_x_start = cubic_bezier(domain[n - 1], base_point[0], anchor1[0], anchor2[0], end_point[0], 1, 0)
        segment_y_start = cubic_bezier(domain[n - 1], base_point[1], anchor1[1], anchor2[1], end_point[1], 1, 0)
        segment_start = [segment_x_start, segment_y_start]

        segment_x_end = cubic_bezier(domain[n], base_point[0], anchor1[0], anchor2[0], end_point[0], 1, 0)
        segment_y_end = cubic_bezier(domain[n], base_point[1], anchor1[1], anchor2[1], end_point[1], 1, 0)
        segment_end = [segment_x_end, segment_y_end]

        length += find_path_length(segment_start, segment_end)

    return length

#find total length of the path and assign the length of each segment to the corresponding path list element
total_path_length = 0

for n in range(2, len(path)):
    if not isinstance(path[n], str):
        continue

    if path[n] == 'L':
        line_length = find_path_length(path[n - 1], path[n + 1])
        path[n] = ('L', line_length)

        total_path_length += line_length

    if path[n] == 'Q':
        quadratic_length = estimate_quadratic_length(path[n - 1], path[n + 1], path[n + 2])
        path[n] = ('Q', quadratic_length)

        total_path_length += quadratic_length

    if path[n] == 'T':
        base_point = path[n - 1]
        first_anchor = [(path[n - 1][0] + (path[n - 1][0] - path[n - 2][0])), (path[n - 1][1] + (path[n - 1][1] - path[n - 2][1]))]
        quadratic_length = estimate_quadratic_length(base_point, first_anchor, path[n + 1])
        path[n] = ('T', quadratic_length)

        total_path_length += quadratic_length

    if path[n] == 'C':
        cubic_length = estimate_cubic_length(path[n - 1], path[n + 1], path[n + 2], path[n + 3])
        path[n] = ('C', cubic_length)

        total_path_length += cubic_length

    if path[n] == 'S':
        base_point = path[n - 1]
        first_anchor = [(path[n - 1][0] + (path[n - 1][0] - path[n - 2][0])), (path[n - 1][1] + (path[n - 1][1] - path[n - 2][1]))]

        cubic_length = estimate_cubic_length(base_point, first_anchor, path[n + 1], path[n + 2])
        path[n] = ('S', cubic_length)

        total_path_length += cubic_length


functionscape = [[],[]]
lower_bound = 0

for i in range(2, len(path)):
    if not isinstance(path[i][0], str):
        continue

    if path[i][0] == 'L':
        p0_x = path[i - 1][0]
        p0_y = path[i - 1][1]
        p1_x = path[i + 1][0]
        p1_y = path[i + 1][1]

        upper_bound = lower_bound + path[i][1]/total_path_length
        functionscape[0].append(((linear_bezier, (p0_x, p1_x)), (lower_bound, upper_bound)))
        functionscape[1].append(((linear_bezier, (p0_y, p1_y)), (lower_bound, upper_bound)))

        lower_bound = upper_bound

    if path[i][0] == 'Q':
        p0_x = path[i - 1][0]
        p0_y = path[i - 1][1]
        p1_x = path[i + 1][0]
        p1_y = path[i + 1][1]
        p2_x = path[i + 2][0]
        p2_y = path[i + 2][1]

        upper_bound = lower_bound + path[i][1]/total_path_length
        functionscape[0].append(((quadratic_bezier, (p0_x, p1_x, p2_x)), (lower_bound, upper_bound)))
        functionscape[1].append(((quadratic_bezier, (p0_y, p1_y, p2_y)), (lower_bound, upper_bound)))

        lower_bound = upper_bound

    if path[i][0] == 'T':
        p0_x = path[i - 1][0]
        p0_y = path[i - 1][1]
        p1_x = path[n - 1][0] + (path[n - 1][0] - path[n - 2][0])
        p1_y = path[n - 1][1] + (path[n - 1][1] - path[n - 2][1])
        p2_x = path[i + 1][0]
        p2_y = path[i + 1][1]

        upper_bound = lower_bound + path[i][1]/total_path_length
        functionscape[0].append(((quadratic_bezier, (p0_x, p1_x, p2_x)), (lower_bound, upper_bound)))
        functionscape[1].append(((quadratic_bezier, (p0_y, p1_y, p2_y)), (lower_bound, upper_bound)))

        lower_bound = upper_bound

    if path[i][0] == 'C':
        p0_x = path[i - 1][0]
        p0_y = path[i - 1][1]
        p1_x = path[i + 1][0]
        p1_y = path[i + 1][1]
        p2_x = path[i + 2][0]
        p2_y = path[i + 2][1]
        p3_x = path[i + 3][0]
        p3_y = path[i + 3][1]

        upper_bound = lower_bound + path[i][1]/total_path_length
        functionscape[0].append(((cubic_bezier, (p0_x, p1_x, p2_x, p3_x)), (lower_bound, upper_bound)))
        functionscape[1].append(((cubic_bezier, (p0_y, p1_y, p2_y, p3_y)), (lower_bound, upper_bound)))

        lower_bound = upper_bound

    if path[i][0] == 'S':
        p0_x = path[i - 1][0]
        p0_y = path[i - 1][1]
        p1_x = path[n - 1][0] + (path[n - 1][0] - path[n - 2][0])
        p1_y = path[n - 1][1] + (path[n - 1][1] - path[n - 2][1])
        p2_x = path[i + 1][0]
        p2_y = path[i + 1][1]
        p3_x = path[i + 2][0]
        p3_y = path[i + 2][1]

        upper_bound = lower_bound + path[i][1]/total_path_length
        functionscape[0].append(((cubic_bezier, (p0_x, p1_x, p2_x, p3_x)), (lower_bound, upper_bound)))
        functionscape[1].append(((cubic_bezier, (p0_y, p1_y, p2_y, p3_y)), (lower_bound, upper_bound)))

        lower_bound = upper_bound

#functionscape = [[((quadratic_bezier, (0.2,0.5,0.8)), (0,0.4)), ((quadratic_bezier, (0.8,0.9,0.1)), (0.4,0.7)), ((quadratic_bezier, (0.1,0.1,0.3)), (0.7,1))],[((quadratic_bezier, (0.2,0.8,0.2)), (0,0.4)), ((quadratic_bezier, (0.2,0.9,0.1)), (0.4,0.7)), ((quadratic_bezier, (0.1,0.5,0.8)), (0.7,0.1))]]

series = 'cos'

plotrange = 1

def linear_by_cos(x,n,L,f,p0,p1,skew,offset):
    return f(x,p0,p1,skew,offset) * math.cos(n * math.pi * x/L)

def linear_by_sin(x,n,L,f,p0,p1,skew,offset):
    return f(x,p0,p1,skew,offset) * math.sin(n * math.pi * x/L)

def quadratic_by_cos(x,n,L,f,p0,p1,p2,skew,offset):
    return f(x,p0,p1,p2,skew,offset) * math.cos(n * math.pi * x/L)

def quadratic_by_sin(x,n,L,f,p0,p1,p2,skew,offset):
    return f(x,p0,p1,p2,skew,offset) * math.sin(n * math.pi * x/L)

def cubic_by_cos(x,n,L,f,p0,p1,p2,p3,skew,offset):
    return f(x,p0,p1,p2,p3,skew,offset) * math.cos(n * math.pi * x/L)

def cubic_by_sin(x,n,L,f,p0,p1,p2,p3,skew,offset):
    return f(x,p0,p1,p2,p3,skew,offset) * math.sin(n * math.pi * x/L)

def half_range_sin_coefficient(L, n, functionscape):
    result = 0
    for i in functionscape:
        if i[0][0] == linear_bezier:
            p0 = i[0][1][0]
            p1 = i[0][1][1]
            skew =  1/(i[1][1] - i[1][0])
            offset = i[1][0]
            anon_func = i[0][0]
            result += quad(linear_by_sin, i[1][0], i[1][1], args=(n,L,anon_func,p0,p1, skew, offset))[0]

        elif i[0][0] == quadratic_bezier:
            p0 = i[0][1][0]
            p1 = i[0][1][1]
            p2 = i[0][1][2]
            skew =  1/(i[1][1] - i[1][0])
            offset = i[1][0]
            anon_func = i[0][0]
            result += quad(quadratic_by_sin, i[1][0], i[1][1], args=(n,L,anon_func,p0,p1,p2, skew, offset))[0]

        elif i[0][0] == cubic_bezier:
            p0 = i[0][1][0]
            p1 = i[0][1][1]
            p2 = i[0][1][2]
            p3 = i[0][1][3]
            skew =  1/(i[1][1] - i[1][0])
            offset = i[1][0]
            anon_func = i[0][0]
            result += quad(cubic_by_sin, i[1][0], i[1][1], args=(n,L,anon_func,p0,p1,p2,p3, skew, offset))[0]

    return 2*result/L

def first_half_range_cos_coefficient(L, functionscape):
    result = 0
    for i in functionscape:
        if i[0][0] == linear_bezier:
            p0 = i[0][1][0]
            p1 = i[0][1][1]
            skew =  1/(i[1][1] - i[1][0])
            offset = i[1][0]
            result += quad(i[0][0], i[1][0], i[1][1], args=(p0,p1, skew, offset))[0]

        elif i[0][0] == quadratic_bezier:
            p0 = i[0][1][0]
            p1 = i[0][1][1]
            p2 = i[0][1][2]
            skew =  1/(i[1][1] - i[1][0])
            offset = i[1][0]
            result += quad(i[0][0], i[1][0], i[1][1], args=(p0,p1,p2, skew, offset))[0]

        elif i[0][0] == cubic_bezier:
            p0 = i[0][1][0]
            p1 = i[0][1][1]
            p2 = i[0][1][2]
            p3 = i[0][1][3]
            skew =  1/(i[1][1] - i[1][0])
            offset = i[1][0]
            result += quad(i[0][0], i[1][0], i[1][1], args=(p0,p1,p2, p3, skew, offset))[0]

    return result/L

def consequent_half_range_cos_coefficient(L, n, functionscape):
    result = 0

    for i in functionscape:
        if i[0][0] == linear_bezier:
            p0 = i[0][1][0]
            p1 = i[0][1][1]
            skew =  1/(i[1][1] - i[1][0])
            offset = i[1][0]
            anon_func = i[0][0]
            result += quad(linear_by_cos, i[1][0], i[1][1], args=(n,L,anon_func,p0,p1, skew, offset))[0]

        elif i[0][0] == quadratic_bezier:
            p0 = i[0][1][0]
            p1 = i[0][1][1]
            p2 = i[0][1][2]
            skew =  1/(i[1][1] - i[1][0])
            offset = i[1][0]
            anon_func = i[0][0]
            result += quad(quadratic_by_cos, i[1][0], i[1][1], args=(n,L,anon_func,p0,p1,p2, skew, offset))[0]

        elif i[0][0] == cubic_bezier:
            p0 = i[0][1][0]
            p1 = i[0][1][1]
            p2 = i[0][1][2]
            p3 = i[0][1][3]
            skew =  1/(i[1][1] - i[1][0])
            offset = i[1][0]
            anon_func = i[0][0]
            result += quad(cubic_by_cos, i[1][0], i[1][1], args=(n,L,anon_func,p0,p1,p2,p3, skew, offset))[0]

    return 2*result/L



if series == 'cos':
    circle_dimensions = [(first_half_range_cos_coefficient(plotrange, functionscape[0]), 0, math.pi/2), (first_half_range_cos_coefficient(plotrange, functionscape[1]), 0, 0)]
elif series == 'sin':
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

def sort_by_first_value(value):
    return value[0]

circle_dimensions.sort(key = sort_by_first_value, reverse = True)

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

tracking =  True

centre = [500,500]
frame_dimensions = (1000,1000)

if tracking == True:
    centre = [500,1600]
    frame_dimensions = (2100,2100)

'''
circle_dimension[0] = radius of circle
circle_dimension[1] = number of times circle rotates around the origin.
circle_dimension[2] = initial offset in radians of the starting point on the circle
'''

if os.path.isdir(os.getcwd() + '/workdir'):
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

        if i == 0:
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
        ctx.arc(finalx,finaly,4,0,2*math.pi)
        ctx.fill()

        ctx.arc(pointsSum(points, 0), pointsSum(points, 1), i[0], 0, 2*math.pi)
        ctx.stroke()

        #if we are on the last circle, fill in the past points and add the current one to the collection
        if i == circle_dimensions[-1]:
            points_collection.append((finalx, finaly))
            fill_past_points(ctx, points_collection)

    radii_position = [centre[0], centre[1]]

    for i in points[1:]:
        ctx.move_to(radii_position[0], radii_position[1])
        radii_position[0] = radii_position[0] + i[0]
        radii_position[1] = radii_position[1] + i[1]
        ctx.line_to(radii_position[0], radii_position[1])

    ctx.stroke()


    if tracking == True:
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

print("completed frame %i" % (FRAMECOUNT))
print("creating gif")
#create gif
images = []
for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave(cwd + '/final.gif', images)
print("removing workdir")
#remove images after gif is created
shutil.rmtree(cwd + '/workdir/')

print('finished')
print('file located at ' + cwd + '/final.gif')
