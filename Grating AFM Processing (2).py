import pySPM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pathlib import Path
import copy
import os
from IPython.display import display
import math as m
from scipy.optimize import curve_fit

# Global settings and user inputs
directory_in_str = str(os.getcwd())
directory = os.fsencode(directory_in_str)

check1 = input('Specify globals? (y/n): ')
if check1.lower() == 'y':
    lengthx = int(input('Length of image in nm: ') or 1800)
    npillars = int(input('Number of full pillars: ') or 2)
    pixels = int(input('Pixels per horizontal line: ') or 512)
    period = float(input('Period of grating: ') or 574.7)
else:
    lengthx, npillars, pixels, period = 1800, 2, 512, 574.7

check2 = input('Specify plot outputs? (y/n): ')
if check2.lower() == 'y':
    oneDAcheck = input('Plot 1D Average Line Out? (y/n): ')
    oneDEcheck = input('Plot 1D Average Error? (y/n): ')
    twoDIcheck = input('Plot 2D image? (y/n): ')
    LERcheck = input('Plot Line Edge Roughness? (y/n): ')
else:
    oneDAcheck, oneDEcheck, twoDIcheck, LERcheck = 'y','y','y','y'

ler_subtraction_check = input('Perform straight line subtraction for LER calculation? (y/n): ')

periodpixellength = int(pixels / lengthx * period)
pixeltonmsf = lengthx / pixels

# Helper functions (from your original code)
def removestreaks(profile):
    return profile.filter_scars_removal(.7,inline=False)

def pillarlocator(profile, npillars, pillaraccuracy=0.9):
    start = list(profile[:int(pillaraccuracy * periodpixellength)]).index(min(profile[:int(pillaraccuracy * periodpixellength)]))
    return [start + periodpixellength * i for i in range(npillars + 1)]

def peaklocator(profile, trenches):
    return [list(profile).index(max(profile[trenches[i]:trenches[i+1]])) for i in range(len(trenches) - 1)]

def trenchlocator(profile, peaks, pillaraccuracy=0.9):
    trenches = [list(profile).index(min(profile[:peaks[0]]))]
    for i in range(len(peaks)):
        try:
            trench = min(profile[peaks[i]:peaks[i+1]])
        except:
            period = periodpixellength if npillars == 1 else peaks[i] - peaks[i-1]
            trench = min(profile[peaks[i]:peaks[i] + int(period * pillaraccuracy)])
        trenches.append(list(profile).index(trench))
        if npillars == 1 or i == len(peaks) - 1:
            break
    if len(trenches) > npillars + 1:
        raise Exception("Too Many Trenches")
    return trenches

def trenchpillarcombiner(profile):
    peaks = peaklocator(profile, pillarlocator(profile, npillars))
    trenches = trenchlocator(profile, peaks)
    return sorted(peaks + trenches)

def flatten(profile, npillars, flatline1delta=0, flatline2delta=0):
    flatline1, flatline2 = flatline1delta, len(profile.pixels) - 1 - flatline2delta
    x1 = trenchlocator(profile.pixels[flatline1], peaklocator(profile.pixels[flatline1], pillarlocator(profile.pixels[flatline1], npillars)))
    x2 = trenchlocator(profile.pixels[flatline2], peaklocator(profile.pixels[flatline2], pillarlocator(profile.pixels[flatline2], npillars)))
    lines = [[x1[i], 0, x2[i], len(profile.pixels) - 1] for i in range(len(x1))]
    return profile.offset(lines)

def fixzero1D(profile):
    return [n - min(profile) for n in profile]

def fixzero2D(profile):
    minimum = np.min(profile.pixels)
    with np.nditer(profile.pixels, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = x - minimum
    return profile
    
def averageprofile(profile, outputerror=False):
    avg, err = profile.mean(0), profile.std(0)
    return (avg, err) if outputerror else avg

def derivativeprofile(profile, n=3):
    return [(profile[i+n] - profile[i-n]) / (2 * n * pixeltonmsf) for i in range(n, len(profile) - n)]

def wallanglecalc(profile, start, end, bp=0.10, tp=0.90):
    segment = profile[start:end+1]
    min_val, max_val = min(segment), max(segment)
    p10, p90 = bp * (max_val - min_val) + min_val, tp * (max_val - min_val) + min_val
    sorted_segment = sorted(segment + [p10, p90])
    p10_idx, p90_idx = sorted_segment.index(p10) - 1, sorted_segment.index(p90) - 2
    opp, adj = abs(sorted_segment[p90_idx] - sorted_segment[p10_idx]), abs(p90_idx - p10_idx)
    return 57.2958 * abs(m.atan(opp / (pixeltonmsf * adj)))

def pillarwidthcalc(profile, start, end, height=0.10):
    segment = profile[start:end]
    peak = max(segment)
    widthpoint = height * (peak - 0.5 * (profile[start] + profile[end])) + min(segment)
    peak_idx = segment.index(peak)
    firstwall, secondwall = segment[:peak_idx], segment[peak_idx:]
    firstwallheight = sorted(firstwall + [widthpoint]).index(widthpoint) - 1
    secondwallheight = len(secondwall) - sorted(secondwall + [widthpoint], reverse=True).index(widthpoint) - 1
    return (secondwallheight + len(firstwall) - firstwallheight) * pixeltonmsf

def assign_pillar_data(data, column_suffix, step=1):
    for i in range(0, npillars * 2, step):
        side = 'Left' if i % 2 == 0 else 'Right'
        pillarn = i // 2 + 1
        df[f'Pillar {pillarn} {side} {column_suffix}'] = data[i // step]

def sigmoid(x, a, b, c, x0):
    y = a + (b - a) / (1 + np.exp(-c * (x - x0)))
    return y

def line_edge(profile):
    x0s = []
    for badrow, row in enumerate(profile):
        rowx0s = []
        pat = trenchpillarcombiner(row)
        for i in range(len(pat)-1):
            segment = row[pat[i]:pat[i+1]]
            guesses = [
                [max(segment), min(segment), 0.1, pat[i]+(pat[i+1]-pat[i])/2],
                [min(segment), max(segment), 0.1, pat[i]+(pat[i+1]-pat[i])/2],
                [min(segment), max(segment), 0.1, pat[i]+(pat[i+1]-pat[i])/1.5],
                [max(segment), min(segment), 0.1, pat[i]+(pat[i+1]-pat[i])/1.5]
            ]
            bounds = (
                [-1000, -1000, 0, 0],
                [1000, 1000, 1, len(row)]
            )
            for guess in guesses:
                try:
                    popt, pcov = curve_fit(sigmoid, list(range(pat[i],pat[i+1])), segment, guess, bounds=bounds)
                    break
                except:
                    if guess == guesses[-1]:

                        pass
                    pass
            rowx0s.append(popt[3])
        x0s.append(rowx0s)
    return x0s

# new stuff
def calculate_ler_with_subtraction(data, subtract_slope=False):
    """
    1)process image to find edge positions
    2)optional leveling by subtracting linear slope
    3)final LER value is calculated as 3X STDEV of edge positions
    """
    processed_data = data.copy()

    if subtract_slope:
        # 1. Get the average profile of the data if subtract_slope=true
        x = np.arange(processed_data.shape[1])
        average_profile = np.mean(processed_data, axis=0)

        # 2. Perform a linear regression, np.polyfit, on the average profile
        coefficients = np.polyfit(x, average_profile, 1)
        average_slope = coefficients[0]
        
        # 3. Create a 2D plane, slope_plane, with this slope to subtract from the data (flattens image/removes tilt)
        height, width = processed_data.shape
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
        slope_plane = average_slope * x_grid
        
        # 4. Subtract the slope plane from the data
        processed_data -= slope_plane

    # 5. Find the line edges from the processed data
    # The line_edge function is called on the processed data. X0 marks the line edge.
    x_values = []
    #try:
    edges = line_edge(processed_data)
    for element in range(len(edges[0])):
        x_values.append([])
    for rowx0s in edges:
        for i, element in enumerate(rowx0s):
            x_values[i].append(element)
    """"
    except:
        # if line_edge fails
        pass
    """
    return np.array(x_values)
    #LER from the processed x_values us calculated in main script

# Main script execution
if __name__ == "__main__":
    
    
    all_results = []
    filestobeprocessed = sum(1 for file in os.listdir(directory) if os.fsdecode(file).endswith(".spm"))
    siteindex = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".spm"):
            Scan = pySPM.Bruker(filename)  
            topo = Scan.get_channel()
            topoE = copy.deepcopy(topo)

            # Modify 2D Data
            topoE = removestreaks(topoE)
            topoE = flatten(topoE, npillars)
            topoE = fixzero2D(topoE)
            
            # Use the new function to get line edge data
            subtract_slope_for_ler = True if ler_subtraction_check.lower() == 'y' else False
            x_values = calculate_ler_with_subtraction(topoE.pixels, subtract_slope=subtract_slope_for_ler)

            
            file_results = {"Filename": filename}

            if x_values.size > 0:
                # Plot 2D image with line edges
                if twoDIcheck.lower() == 'y':
                    fig, ax = plt.subplots()
                    ax.imshow(topoE.pixels)
                    if LERcheck.lower() == 'y':
                        y = range(x_values.shape[1])
                        for x_series in x_values:
                            ax.plot(x_series, y, linewidth=1.5, color='red')
                    plt.savefig(f'{filename.split('.')[0]} {filename.split('.')[1]}.png')
                    mpl.pyplot.close()

                # Plot 2D line edge only
                if LERcheck.lower() == 'y':
                    y = range(x_values.shape[1])
                    fig, ax = plt.subplots()
                    for x_series in x_values:
                        ax.plot(x_series, y, linewidth=2.0)
                    plt.savefig(f'{filename.split('.')[0]} {filename.split('.')[1]} Line Edge Profile.png')
                    mpl.pyplot.close()
                
                # Calculate LER from the processed x_values and add to the file_results dictionary
                for i in range(x_values.shape[0]):
                    file_results[f'LER Edge {i+1}'] = 3 * np.std(x_values[i])
                    print("Ler calculated")
            # Modify 1D average profile
            averageprofileoutput = averageprofile(topoE.pixels, outputerror=True)
            averageprofilelist = fixzero1D(averageprofileoutput[0])
            averageprofileerror = averageprofileoutput[1]
            derivativeprofilelist = derivativeprofile(averageprofilelist)

            l = list(averageprofilelist)
            d = list(map(lambda n: abs(n), derivativeprofilelist))
            importantpoints = trenchpillarcombiner(l)
            p = importantpoints

        
            for i in range(npillars * 2):
                if i+1 < len(p):
                    file_results[f'Pillar {i//2+1} {"Left" if i%2==0 else "Right"} Height'] = abs(l[p[i+1]] - l[p[i]])
                    file_results[f'Pillar {i//2+1} {"Left" if i%2==0 else "Right"} Derivative Angle'] = 57.2958 * m.atan(max(d[p[i]:p[i+1]]))
                    file_results[f'Pillar {i//2+1} {"Left" if i%2==0 else "Right"} Wall Angle'] = wallanglecalc(l, p[i], p[i+1])
                if i % 2 == 0 and i+2 < len(p):
                    width = pillarwidthcalc(l, p[i], p[i+2])
                    file_results[f'Pillar {i//2+1} Width'] = width
                    file_results[f'Pillar {i//2+1} Duty Cycle'] = width / period
            
            file_results['average error']=np.mean(averageprofileerror)
            
            
            
            all_results.append(file_results)

            print(f'Processed Files: {siteindex+1}/{filestobeprocessed}')
            siteindex += 1

    # dataframe
    if all_results:
        df = pd.DataFrame(all_results)
        
        
        foldername = str(os.getcwd()).split('\\')[-1]
        writer = pd.ExcelWriter(f"{foldername} Pillar Characterization.xlsx", engine='xlsxwriter')
        df.to_excel(writer, sheet_name="Pillars", index=False)
        writer._save()
    else:
        print("No .spm files were found or processed.")

    print('Processing Completed')
    input('Press enter to close')