# Fourier Series Visualization

A python project to create a visual application of Fourier Series to trace out a path in the 2D plane, using two separate parameterised series for the x and y axis.

![](example.gif)

This file processes the path attribute from an .svg file, that is to say, a single continuous line composed of straight lines, quadratic and cubic bezier curves.
It then decomposes this path into x and y parameters as a function of time, such that movement along the path is at a constant speed the whole way.
Finally, it calculates the Fourier series for each parameter, and converges them into one set of circles to draw the desired path.

### How to apply it

Create an svg file with a desired path drawn, and copy the 'd' element of the path attribute. This will look something like:
```<path d="M 250 119 L 110 119 . . ."></path>```
Then simply run Untitled.py and pass in the string value.
An example path:
`M 260.66302490234375 100.11151123046875 L 90.58003997802734 208.88551330566406 L 179.5769500732422 208.88551330566406 L 96.51316833496094 313.7041015625 L 177.5992431640625 313.7041015625 L 96.51316833496094 438.2998046875 L 230.9973907470703 436.32208251953125 L 230.9973907470703 556.9623413085938 L 280.44012451171875 558.9400634765625 L 278.46240234375 438.2998046875 L 381.30328369140625 438.2998046875 L 327.9051513671875 315.68182373046875 L 379.3255920410156 315.68182373046875 L 329.8828430175781 212.8409423828125 L 383.281005859375 210.8632354736328 L 272.529296875 100.11151123046875`
