from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# colors defined by matplotlib
#https://matplotlib.org/stable/gallery/color/named_colors.html
blue_1 = mcolors.TABLEAU_COLORS['tab:blue']
blue_2 = mcolors.CSS4_COLORS['cornflowerblue']
blue_3 = mcolors.CSS4_COLORS['navy']
blue_4 = mcolors.CSS4_COLORS['lightskyblue']
blue_5 = mcolors.CSS4_COLORS['royalblue']
blue_6 = mcolors.CSS4_COLORS['mediumblue']
blue_7 = mcolors.CSS4_COLORS['blue']
green_1 = mcolors.TABLEAU_COLORS['tab:green']
green_2 = mcolors.CSS4_COLORS['yellowgreen']
green_3 = mcolors.CSS4_COLORS['olivedrab']
green_4 = mcolors.CSS4_COLORS['darkgreen']
green_5 = mcolors.CSS4_COLORS['seagreen']
green_6 = mcolors.CSS4_COLORS['green']
olive = mcolors.CSS4_COLORS['olive']
red_1 = mcolors.TABLEAU_COLORS['tab:red']
red_2 = mcolors.CSS4_COLORS['firebrick']
red_3 = mcolors.CSS4_COLORS['darkred']
red_4 = mcolors.CSS4_COLORS['salmon']
gray_1 = mcolors.CSS4_COLORS['dimgray']
gray_2 = mcolors.CSS4_COLORS['darkgrey']
gray_3 = mcolors.CSS4_COLORS['gainsboro']
black = mcolors.CSS4_COLORS['black']
orange_1 = mcolors.TABLEAU_COLORS['tab:orange']
orange_2 = mcolors.CSS4_COLORS['darkorange']
orange_3 = mcolors.CSS4_COLORS['orange']
gold = mcolors.CSS4_COLORS['gold']
yellow_1 = mcolors.CSS4_COLORS['yellow']
purple_1 = mcolors.CSS4_COLORS['purple']
purple_2 = mcolors.CSS4_COLORS['indigo']
blue_violet = mcolors.CSS4_COLORS['blueviolet']
dark_slate_blue = mcolors.CSS4_COLORS['darkslateblue']
sienna = mcolors.CSS4_COLORS['sienna']
slategrey = mcolors.CSS4_COLORS['slategrey']
darkslategrey = mcolors.CSS4_COLORS['darkslategrey']
peru = mcolors.CSS4_COLORS['peru']
darkgoldenrod = mcolors.CSS4_COLORS['darkgoldenrod']
indigo = mcolors.CSS4_COLORS['indigo']


COLOR_CAMERA = blue_5
COLOR_MESH = green_4
COLOR_STITCHES_3D_MAP = indigo

bgr_black = (0, 0, 0)
bgr_white = (255, 255, 255)
bgr_blue = (255, 0, 0)
bgr_green = (0, 255, 0)
bgr_red = (0, 0, 255)
bgr_lilac = (189, 130, 188)
bgr_yellow = (0, 255, 255)



def plot_colortable(colors, title, sort_colors=True, emptycols=0):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)
    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')
        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )
    return fig, colors

if __name__ == '__main__':
     # execute only if run as a script
    plot_colortable(mcolors.BASE_COLORS, "Base Colors",
                    sort_colors=False, emptycols=1)
    plot_colortable(mcolors.TABLEAU_COLORS, "Tableau Palette",
                    sort_colors=False, emptycols=2)
    plot_colortable(mcolors.CSS4_COLORS, "CSS Colors")
    # Optionally plot the XKCD colors (Caution: will produce large figure)
    xkcd_fig = plot_colortable(mcolors.XKCD_COLORS, "XKCD Colors")
    # xkcd_fig = plot_colortable(mcolors.XKCD_COLORS, "XKCD Colors")
    # xkcd_fig.savefig("XKCD_Colors.png")

