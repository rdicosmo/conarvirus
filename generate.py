#!/usr/bin/python3

import matplotlib.pyplot as plt
from datetime import date,timedelta
import numpy as np
import csv
import os


death_threshold = 10

regions = {
    'Asia': ['Heilongjiang(China)', 'Henan(China)', 'Hubei(China)', 'India', 'Indonesia', 'Japan', 'Korea, South', 'Malaysia', 'Philippines', 'Beijing(China)', 'Guangdong(China)', 'Shandong(China)', 'Iran', 'Iraq', 'Lebanon', 'Pakistan', 'Bangladesh', 'Anhui(China)', 'Chongqing(China)', 'Hainan(China)', 'Hebei(China)', 'Shanghai(China)', 'Thailand'],
    'Europe': ['Albania', 'Austria', 'Belgium', 'Denmark', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'San Marino', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'United Kingdom', 'Diamond Princess(Boat)', 'Czechia', 'Finland', 'Luxembourg', 'Slovenia', 'Iceland', 'Serbia', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Lithuania', 'Ukraine'],
    'North America': ['British Columbia(Canada)','Ontario(Canada)', 'Dominican Republic', 'US', 'Quebec(Canada)', 'Panama', 'Peru' ],
    'South America': ['Argentina', 'Brazil', 'Ecuador', 'Mexico', 'Chile', 'Colombia'],
    'Australia': ['New South Wales(Australia)'],
    'Africa': ['Burkina Faso', 'Tunisia', 'Algeria', 'Egypt', 'Israel', 'Congo (Kinshasa)', 'Ghana', 'Morocco'],
    'World': ['Asia', 'Europe', 'North America', 'South America', 'Africa'] #, 'Australia']
}    


#############################
# Loading and filtering data

def reformat_location(location):

    l1,l2=location.split("/")
    if l2=="" or l1==l2:
        return l1
    else:
        return l2+"("+l1+")"


def get_data_from_files():

    files = [ "time_series_covid19_deaths_global.csv", "time_series_covid19_confirmed_global.csv", "time_series_covid19_recovered_global.csv" ] # files where to fetch data

    data2 = dict()
    i=0
    for key in keys:
        data2[key]=dict()
        with open(files[i]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    days = row[4:]
                else:
                    location = row[1]+"/"+row[0]
                    p = reformat_location(location)
                    if p=="Diamond Princess":
                        p="Diamond Princess(Boat)"
                    data = [ int(row[j]) for j in range(4,len(row))]
                    # store data
                    data2[key][p] = data                            
                line_count += 1
        i += 1

    #data2["Total"] = dict()
    
    return(data2, days)


def filter_by_dates(data, days, begin, end):

    data2 = dict()
    a = days.index(begin)
    b = days.index(end)
    for k in data:
        data2[k] = dict()
        for p in data[k]:
            data2[k][p] = data[k][p][a:b+1]
            
    return(data2, days[a:b+1])


def filter_by_var_threshold(data, var, threshold=death_threshold, sync=False):

    data2, xr = dict(), dict()
    for k in data:
        data2[k], xr[k] = dict(), dict()

    for p in data[var]:
        if p in data[var]:
            d = data[var][p]
            #print(p,d)
            if max(d) >= threshold:
                j = list(map(lambda i: i>=threshold, d)).index(True)
                if j<len(d)-1:
                    for k in data:
                        if p in data[k]:
                            d2 = data[k][p][j:]
                            if sync:
                                data2[k][p] = d2
                                xr[k][p] = range( len(d2) )
                            else:
                                data2[k][p] = data[k][p]
                                xr[k][p] = range( -len(data2[k][p])+1, 1 )

    return(data2, xr)


def filter_by_regions(data, l, bool=True):

    data2 = dict()
    for k in data:
        data2[k] = dict()
        for p in data[k]:
            if (p in l)==bool:
                data2[k][p] = data[k][p]

    return data2


def add_regions(data):

    for name in regions:
        lst=regions[name]
        for k in data:
            l = []
            for p in lst:
                if p in data[k]:
                    l.append(data[k][p])
            data[k][name] = [sum(x) for x in zip(*l)]

    
#####################################
# Data smoothing and differentiation

def diff(f): # discrete differentiation
    return( [ 0.0 ] + [ f[j]-f[j-1]  for j in range(1, len(f)) ] )

def smooth(y, smooth_parameter, c=3.0 ):  # local linear smoothing
    y2 = y.copy()
    for i in range(smooth_parameter):
        y2 = [ y2[0] ] + [ (y2[j-1] + (c-2)*y2[j] + y2[j+1])/c for j in range(1,len(y2)-1) ] + [ y2[-1] ]
    return y2

def smooth_speed_acceleration(data, smooth_parameter):
    #
    dg, gs, dgs, ddgs = dict(), dict(), dict(), dict()
    for key in keys:

        dg[key], gs[key], dgs[key], ddgs[key] = dict(), dict(), dict(), dict()
        for f in data[key]:
            dg[key][f] = diff( data[key][f] )
            
            gs[key][f] =  list(map(lambda x: np.exp(x)-1, 
                              smooth(
                                  list(map(lambda x: np.log(1+x), data[key][f])), smooth_parameter   # smoothing of the log(1+x)-transformed function
                              )
            ))
            dgs[key][f] = diff( gs[key][f] )   # speed
            ddgs[key][f] = diff( dgs[key][f] ) # acceleration
            
    return dg, gs,dgs,ddgs


#########################
# Functions to plot data

def style_args(data, region_list):   # define style parameters for plot and text for each region/country

    colors = [ "black", "grey", "indianred", "darkred", "tomato", "peru", "olivedrab", "cadetblue", "darkblue", "crimson", "darkmagenta", "red", "blue", "green" ]
    lc = len(colors)
    linestyles = [ "-", "--", ":", "-." ]

    # sort countries by max number of deaths
    
    arg_plot = dict()
    k = 0

    for phase in [0,1]:
        for f in region_list:
            if (phase==0 and f in regions) or (phase==1 and f not in regions):
                arg_plot[f] = dict()
                arg_plot[f]["color"] = colors[k%lc]
                arg_plot[f]["linestyle"] = linestyles[int(k/lc)]
                k+=1
                
    return arg_plot


def plot(region_list, focus, xr, g, dg, gs, dgs, ddgs, arg_plot, tmax, size=3, log=True):

    lk=len(keys)

    fig = plt.figure(figsize=(size*4,size*lk))
    
    for i in range(lk):
        key = keys[i]
        l = 0
        for (title, fc, yscale, ylim) in [ ("Total number of "+key, gs, ('log', {}), 10.0),
                                                ("Number of "+key+" by day $\\left(\\frac{\\Delta "+key.replace(" ","\\_")+"}{\\Delta t}\\right)$", dgs, ('log', {}), 1.0),
                                                ("Acceleration of the number of "+key+" $\\left(\\frac{\Delta^2 "+key.replace(" ","\\_")+"}{(\Delta t)^2}\\right)$", ddgs, ('symlog',{'linthreshy': 10.0}), 0)
                                                ]: 

            ax = plt.subplot( lk, 3, 3*i + l+1 )
            plt.title(title, fontsize=int(size*2.7))

            for f in fc[key]:

                color = arg_plot[f]["color"]  # same color and width as graph
                if f in focus:
                    lw=2
                else:
                    lw=1
                    
                if l in [0] and not log:    
                    
                    plt.plot( xr[key][f], [ g[key][f], dg[key][f] ][l], "+", mew=lw/2.0, ms=lw*2.5, color=color )
                    
                if len(xr[key][f]) > l:
                    
                    plt.plot( xr[key][f][l:], fc[key][f][l:], lw=lw, **arg_plot[f] )
                    
                    if fc[key][f][-1] > ylim or yscale[0]!="log" or not log:
                        if f in focus:
                            arg_text={"bbox": {'facecolor':'white', 'alpha':0.5, 'boxstyle':"round", 'edgecolor':color},
                                      "fontsize": 7,
                                      "color":color
                            }
                        else:
                            arg_text={"fontsize": 7,
                                      "color":color
                            }
                            
                        plt.text( xr[key][f][-1] , fc[key][f][-1], f, **arg_text )


            if log:
                plt.yscale(yscale[0],**yscale[1])
                if ylim!=0:
                    plt.ylim(ylim)
                
            x1,_ = ax.get_xlim()
            x2 = x1 + tmax - l
            plt.xlim(x1,x2+(x2-x1)/5.0)
            
            plt.grid(True,which="both")
            l+=1
            
    plt.tight_layout()    
    plt.subplots_adjust(top=0.92)
    
    return(fig)


#########################################
#### Main function


# Common part

def check_for_countries(threshold=death_threshold):

    data, days = get_data_from_files()
    data, xr = filter_by_var_threshold(data, "deaths", threshold, sync = False)
    
    print("\nWARNING: New countries to add to continents ?")
    for p in data["deaths"]:
        flag=False
        for r in regions:
            if p in regions[r]:
                flag=True
                break
        if not flag:
            print(p)

    print()
    
    
def prepare_data(begin="", end=""):
    
    data, days = get_data_from_files()  # get days
    if begin=="":
        begin = days[0]
    if end=="":
        end = days[-1]

    data, days = filter_by_dates(data, days, begin, end)  # get data filtered by filter by date
    add_regions(data)   # add all regions
    data, xr = filter_by_var_threshold(data, 'deaths', death_threshold, sync=False) # filter by number of deaths
    
    # sort the regions/coutries by number of deaths
    d = data["deaths"] 
    c = [x for x in d]
    mv = np.array([ max(d[x]) for x in d ])
    bests = list( mv.argsort()[:][::-1] )
    region_list = [ c[i] for i in bests ]

    # associate line colors and styles
    arg_plot= style_args(data, region_list)
    
    return data, days, xr, arg_plot

    

def prepare_graph(data, days, category, sync, log=False):

    if category=="top10":  # top10

        title = "Dynamics of Covid-19 in the 10 countries with the most deaths"
        data2 = filter_by_regions(data, ['World']+regions['World'], bool=False ) # remove regions
        d = data2["deaths"]
        c = [x for x in d]
        mv = np.array([ max(d[x]) for x in d ])
        bests = list( mv.argsort()[-10:][::-1] )
        
        region_list = [ c[i] for i in bests ]
        focus = []
        
    elif category in regions:    
            
        title = "Dynamics of Covid-19"
        if category!="World":
            title+=" in "+category

        if not log:
            region_list = [i for i in regions[category]]+[category]
            focus=[category]
        else:
            region_list = [i for i in regions[category]]
            focus=[]
            
    else:
        
        print("*** Wrong category:", category)
        exit(1)

    data2 = filter_by_regions(data, region_list)
        
    # copy (with potential synchronization)
    data2, xr2 = filter_by_var_threshold(data2, 'deaths', death_threshold, sync=sync) # filter by number of deaths
    
    if sync:
        title+=", day 0: first day such that $deaths \\geq "+str(death_threshold)+"$"
    title+=" (data from JHU CSSE on "+str(days[-1])+")"      

    tmax = max([ len(data2['deaths'][x]) for x in data2['deaths'] ] )
    
    return data2, xr2, region_list, focus, title, tmax


def get_filename(category,sync):
    filename=category.replace(' ','_')
    if sync:
        filename+="_sync"
    return(filename)


def curves(category, filename="", begin="", end="", smooth_parameter=15, size=3, sync=False, log=True):

    filename=get_filename(category, sync)
    
    print("Graph for "+category+", sync=",sync, "=>",filename)
        
    data, days, xr, arg_plot = prepare_data(begin=begin, end=end)
    
    data2, xr2, region_list, focus, title, tmax = prepare_graph(data, days, category=category, sync=sync, log=log)
    dg, gs, dgs, ddgs = smooth_speed_acceleration(data2, smooth_parameter)
    fig = plot(region_list, focus, xr2, data2, dg, gs, dgs, ddgs, arg_plot, tmax, size=size, log=log)
    plt.suptitle(title)
    plt.savefig(filename+".png", dpi=dpi)
    plt.close('all')
    

def regularise(category, filename="", begin="", end="", smooth_parameter=15, size=3, sync=False, log=True):

    filename=get_filename(category, sync)
    
    print("Animation for "+category+" and several smoothing parameters, sync=",sync,"=>",filename)

    data, days, xr, tmax, arg_plot = prepare_data(begin=begin, end=end)
    data2, xr2, region_list, focus, title, tmax = prepare_graph(data, days, category=category, sync=sync)

    smooth_parameters = range(smooth_parameter + 1)
    ns = len(smooth_parameters)
    for n in range(ns):
        print("Smooth parameter:",smooth_parameters[n])
        dg, gs, dgs, ddgs = smooth_speed_acceleration(data, n)
        fig = plot(xr, data, gs, dgs, ddgs, arg_plot, tmax, size=size, log=log)
        plt.suptitle(title+", smooth="+str(n))
        plt.savefig("fig/"+filename+"_%02d.png"%n, dpi=dpi)
        plt.close('all')

    anim_command = "convert -verbose -delay 10 -loop 0 "        
    lns = list(range( ns )) + [ns-1]*4 + list(range(ns-1,-1,-1)) + [0]*4
    src = " ".join(["fig/"+filename+"_%02d.png"%i for i in lns ])
    print("Generate animations from : "+src)
    #os.system( anim_command+src+" "+filename+".mp4" )
    os.system( anim_command+src+" "+filename+".gif" )

    
def evolution(category, begin="", end="", smooth_parameter=15, size=3, sync=False, log=True):
    
    filename=get_filename(category, sync)+"_evol"

    print("Animation for "+category+" and several dates, sync=",sync,"=>",filename)

    data, days, xr, arg_plot = prepare_data(begin=begin, end=end)
    data2, xr2, region_list, focus, title, tmax = prepare_graph(data, days, category=category, sync=sync, log=log)
    dg, gs, dgs, ddgs = smooth_speed_acceleration(data, smooth_parameter)
        
    for t in range(0,tmax):
        
        print("t=",t,"day=",days[t])

        # Crop all data 
        data3, dg3, xr3, gs3, dgs3, ddgs3 = dict(), dict(), dict(), dict(), dict(), dict() 
        for key in keys:
            data3[key], dg3[key], xr3[key], gs3[key], dgs3[key], ddgs3[key] = dict(), dict(), dict(), dict(), dict(), dict()
            for f in data2[key]:

                z = max(0, len(data2[key][f]) - tmax + t + 1)
                data3[key][f] = data2[key][f][0:z]
                dg3[key][f] = dg[key][f][0:z]
                xr3[key][f] = xr[key][f][0:z]
                gs3[key][f] = gs[key][f][0:z]
                dgs3[key][f] = dgs[key][f][0:z]
                ddgs3[key][f] = ddgs[key][f][0:z]
                
        fig = plot(region_list, focus, xr3, data3, dg3, gs3, dgs3, ddgs3, arg_plot, tmax, size=size, log=log)
        plt.suptitle(title+", day="+str(days[t]))
        plt.savefig("fig/"+filename+"_%02d.png"%t, dpi=dpi)
        plt.close('all')

    anim_command = "convert -verbose -delay 10 -loop 0 "
    ltmax = list( range( 3,tmax-1 ) ) 
    src = " ".join([ "fig/"+filename+"_%02d.png"%i for i in ltmax ])
    print("Generate animations : "+src)
    #os.system( anim_command+src+" -delay 300 fig/"+filename+"_%02d.png"%(tmax-1)+" "+filename+".mp4" )
    os.system( anim_command+src+" -delay 300 fig/"+filename+"_%02d.png"%(tmax-1)+" "+filename+".gif" )


#########################
# Main global parameters

keys = [ "deaths", "confirmed cases", "recovered cases" ] # data to show

check_for_countries(5)

dpi = 90 # graph quality (for png/mp4)

for (region, begin) in [
        ('World',''),
        ('top10',''),
        ('Europe','1/29/20'),
        ('Asia',''),
        ('North America','2/27/20'),
        ('South America','3/15/20'),
        ('Africa','3/9/20')
#        ('Australia',''),        
        ]:
    curves(region,log=True, begin=begin)                 # log graph...
    curves(region,log=True, sync=True)                   # log graph synchronized
    evolution(region, log=False, begin=begin)            # and non-log animation


