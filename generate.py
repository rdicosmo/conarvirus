#!/usr/bin/python3

import matplotlib.pyplot as plt
from datetime import date,timedelta,datetime
import numpy as np
import csv
import os


keys = [ "deaths", "confirmed cases", "recovered cases" ] # data to process
death_threshold = 10


regions = {
    'Asia': ['Heilongjiang(China)', 'Henan(China)', 'Hubei(China)', 'India', 'Indonesia', 'Japan', 'Korea, South', 'Malaysia', 'Philippines', 'Beijing(China)', 'Guangdong(China)', 'Shandong(China)', 'Iran', 'Iraq', 'Lebanon', 'Pakistan', 'Bangladesh', 'Anhui(China)', 'Chongqing(China)', 'Hainan(China)', 'Hebei(China)', 'Shanghai(China)', 'Thailand', 'Turkey'],
    'Europe': ['Albania', 'Austria', 'Belgium', 'Denmark', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'San Marino', 'Spain', 'Sweden', 'Switzerland',  'United Kingdom', 'Diamond Princess(Boat)', 'Czechia', 'Finland', 'Luxembourg', 'Slovenia', 'Iceland', 'Serbia', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Lithuania', 'Ukraine'],
    'North America': ['British Columbia(Canada)','Ontario(Canada)', 'Dominican Republic', 'US', 'Quebec(Canada)', 'Panama', 'Peru' ],
    'South America': ['Argentina', 'Brazil', 'Ecuador', 'Mexico', 'Chile', 'Colombia'],
    'Australia': ['New South Wales(Australia)'],
    'Africa': ['Burkina Faso', 'Tunisia', 'Algeria', 'Egypt', 'Israel', 'Congo (Kinshasa)', 'Ghana', 'Morocco'],
    'World': ['Asia', 'Europe', 'North America', 'South America', 'Africa', 'Australia']
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


def filter_by_var_threshold(data, var, threshold=death_threshold):

    data2 = dict()
    for k in data:
        data2[k] = dict()

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
                            data2[k][p] = d2
                
    return(data2)


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

def smooth(y, sm, c=3.0 ):  # local linear smoothing
    y2 = y.copy()
    for i in range(sm):
        y2 = [ y2[0] ] + [ (y2[j-1] + (c-2)*y2[j] + y2[j+1])/c for j in range(1,len(y2)-1) ] + [ y2[-1] ]
    return y2

def compute_fcts(data, sm):
    #
    dg, gs, dgs, ddgs, gr = dict(), dict(), dict(), dict(), dict()
    for key in keys:

        # calcul des dérivées (avec régularisation)
        dg[key], gs[key], dgs[key], ddgs[key], gr[key] = dict(), dict(), dict(), dict(), dict()
        
        for f in data[key]:
            dg[key][f] = diff( data[key][f] )
            
            gs[key][f] =  list(map(lambda x: np.exp(x)-1, 
                              smooth(
                                  list(map(lambda x: np.log(1+x), data[key][f])), sm   # smoothing of the log(1+x)-transformed function
                              )
            ))
            dgs[key][f] = diff( gs[key][f] )   # speed
            ddgs[key][f] = diff( dgs[key][f] ) # acceleration


            if key=='confirmed cases':
                th=1
            else:
                th=1
            gr[key][f]=[ np.nan ]
            for i in range(1,len(gs[key][f])):
                if gs[key][f][i-1] < th:
                    gr[key][f].append( np.nan )
                else:
                    gr[key][f].append( gs[key][f][i]/gs[key][f][i-1]-1.0 )
                            
    return [data, dg, gs, dgs, ddgs, gr]


#########################
# Functions to plot data

def style_args(region_list):   # define style parameters for plot and text for each region/country

    colors = [ "black", "grey", "indianred", "darkred", "tomato", "peru", "olivedrab", "cadetblue", "darkblue", "crimson", "darkmagenta", "darkseagreen", "tan", "darkgoldenrod", "plum", "steelblue", "darkkhaki" ]
    lc = len(colors)
    linestyles = [ "-", "--", ":", "-." ]

    # sort countries by max number of deaths
    
    arg_plot = dict()
    k = 0

    for phase in [0,1]: # phase 0: regions, phase 1: sub-regions, countries
        for f in region_list:
            if (phase==0 and f in regions) or (phase==1 and f not in regions):
                arg_plot[f] = dict()
                arg_plot[f]["color"] = colors[k%lc]
                arg_plot[f]["linestyle"] = linestyles[int(k/lc)]
                k+=1
                
    return arg_plot


def get_dx_xr_color_lw(nl, f, sync, lmax, arg_plot, focus):
    if (nl in [1,3] and len(f)>1):
        dx=1
    elif (nl==2 and len(f)>2):
        dx=2
    else:
        dx=0
    if sync:
        xr = range(0, lmax)
    else:
        xr = range(-lmax+1, 1)
    color = arg_plot[f]["color"]  # color and width of graph/text
    if f in focus:
        lw=2
    else:
        lw=1
    return dx, xr, color, lw


def plot(region_list, what_to_plot, focus, fcts, arg_plot, size=5, log=True, sync=False):

    [ g, dg, gs, dgs, ddgs, gr ]=fcts
    lk = len(what_to_plot[0])
    lg = len(what_to_plot[1])

    fig = plt.figure(figsize=(size*1.33*lg, size*lk))

    for i in range(lk):
        
        key = what_to_plot[0][i]
        nkey = keys.index(key)
        
        for l in range(len(what_to_plot[1])):

            nl = what_to_plot[1][l]
            
            key2 = key.replace(" ","\\_")
            (title, fc, yscale, ylim)= [ ("Total number of "+key, gs, ('log', {}), [10.0, 100.0, 10.0]),
                                         ("Speed (nb of "+key+" by day) $\\left(\\frac{\\Delta "+key2+"}{\\Delta t}\\right)$", dgs, ('log', {}), [1.0, 10.0, 1.0]),
                                         ("Acceleration of the nb of "+key+" $\\left(\\frac{\Delta^2 "+key2+"}{(\Delta t)^2}\\right)$", ddgs, ('symlog',{'linthreshy': 10.0}), [0,0,0]),
                                         ("Daily growth rate of "+key+" $\\left(\\frac{\\Delta "+key2+"}{"+key2+"}\\right)$" , gr, ('', {}), [0,0,0])
            ][ nl ]

            ax = plt.subplot( lk, lg, lg*i + l+1 )
            plt.title(title, fontsize=int(size*2.7))

            max_len = get_max_len(fc[key])

            
            for f in fc[key]:

                myf = fc[key][f]
                
                dx, xr, color, lw = get_dx_xr_color_lw( nl, f, sync, len(myf), arg_plot, focus )
                    
                plt.plot( xr[dx:], myf[dx:], lw=lw, **arg_plot[f] )

                if len(myf)>0:
                    if nl==3 or myf[-1] > ylim[ nkey ] or yscale[0]!='log' or not log:
                        if f in focus:
                            arg_text={"bbox": {'facecolor':'white', 'alpha':0.5, 'boxstyle':"round", 'edgecolor':color},
                                      "fontsize": 9,
                                      "color":color
                            }
                        else:
                            arg_text={"fontsize": 9,
                                      "color":color
                            }        
                        plt.text( xr[-1] , myf[-1], f, **arg_text )


            if sync:
                x1,x2 = 0, max_len
            else:
                x1,x2 = - max_len, 0
            
            if nl==3: # gr
                for j in [2,3,4,5,6,7,10,20,30]:
                    z = pow(2.0,1.0/j)-1
                    plt.plot([x1, x2+(x2-x1)/5.0], [z]*2,"--",color='lightgrey', lw=1, zorder=0)
                    plt.text(x1,z,"$\\times 2$ in %d days"%j, fontsize=8, color="grey", zorder=0)
                plt.ylim(0, 0.5)
                
            elif log:
                
                plt.yscale(yscale[0],**yscale[1])
                if yscale[0]=='log':
                    plt.ylim(bottom = ylim[ nkey ])

            plt.grid(True,which="both")
            plt.xlim(x1,x2+(x2-x1)/5.0) # room for country name

            yr = ax.get_ylim() # get y upper limit before potentially plotting data
           
        
            if  nl in [0, 1]:#  when to plot real data

                for f in fc[key]:

                    myf = fc[key][f]
                    
                    dx, xr, color, lw = get_dx_xr_color_lw( nl, f, sync, len(myf), arg_plot, focus )
                    
                    plt.plot( xr[dx:], [ g[key][f][dx:], dg[key][f][dx:] ][ nl ], "+", mew=lw/2.0, ms=lw*2.5, color=color )
                
            plt.ylim(yr)
                    
            
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    return(fig)


#########################################
#### Main function


# Common part

def check_for_countries(threshold=death_threshold):

    data, days = get_data_from_files()
    data = filter_by_var_threshold(data, "deaths", threshold)
    
    l=[]
    for p in data["deaths"]:
        flag=False
        for r in regions:
            if p in regions[r]:
                flag=True
                break
        if not flag:
            l.append(p)

    if l!=[]:
        print("\nWARNING: New countries to add to continents:")
        for x in l:
            print(x)
        exit(1)
    
    
def prepare_data(begin="", end=""):
    
    data, days = get_data_from_files()  # get days
    if begin=="":
        begin = days[0]
    if end=="":
        end = days[-1]

    data, days = filter_by_dates(data, days, begin, end)  # get data filtered by filter by date
    add_regions(data)   # add all regions

    data = filter_by_var_threshold(data, 'deaths', death_threshold) # filter by number of deaths
    
    # sort the regions/coutries by number of deaths
    d = data["deaths"] 
    c = [x for x in d]
    mv = np.array([ max(d[x]) for x in d ])
    bests = list( mv.argsort()[:][::-1] )
    region_list = [ c[i] for i in bests ]

    # associate line colors and styles to countries
    arg_plot= style_args(region_list)
    
    return data, days, arg_plot

    

def prepare_graph(data, days, category):

    if category=="top10":  # top10

        title = "10 countries with the most deaths"
        data2 = filter_by_regions(data, ['World']+regions['World'], bool=False ) # remove regions
        d = data2["deaths"]
        c = [x for x in d]
        mv = np.array([ max(d[x]) for x in d ])
        bests = list( mv.argsort()[-10:][::-1] )
        
        region_list = [ c[i] for i in bests ]
        focus = []
        
    elif category in regions:    
                    
        title=category

        region_list = [i for i in regions[category]]+[category]
        focus=[category]
            
    else:
        
        print("*** Wrong category:", category)
        exit(1)

    data2 = filter_by_regions(data, region_list)
    
    if sync:
        title+=", day 0=1st day s.t. $deaths \\geq "+str(death_threshold)+"$ (data from JHU CSSE "+str(days[-1])+")"
    else:
        title+=" (data from JHU CSSE, day 0="+str(days[-1])+")"      
    
    return data2, region_list, focus, title


def get_max_len(fs): # for a dictionary of functions
    return max( [ len(fs[x]) for x in fs ] )


def get_filename(category,sync, log, what_to_plot):

    filename=category.replace(' ','_') + ["","_sync"][int(sync)] + ["","_log"][int(log)] + "_"
    for i in what_to_plot[0]:
        filename+=i[0]
    filename+="_"
    for i in what_to_plot[1]:
        filename+=str(i)
    return(filename)


def curves(category, what_to_plot=(keys, [0,1,2]), filename="", begin="", end="", sm=0, size=5, sync=False, log=True):

    filename=get_filename(category, sync, log, what_to_plot)
    
    print("Graph for "+category+", sync=",sync, "=>",filename)
        
    data, days, arg_plot = prepare_data(begin=begin, end=end)
    
    data2, region_list, focus, title = prepare_graph(data, days, category=category)
    fcts = compute_fcts(data2, sm)
    fig = plot(region_list, what_to_plot, focus, fcts, arg_plot, size=size, log=log, sync=sync)
    plt.suptitle(title, fontsize=16)
    plt.savefig("./fig/"+filename+".png", dpi=dpi)
    plt.close('all')
    
    
def animation(category, what_to_plot=(keys, [0,1,2]), begin="", end="", sm=0, size=5, log=False):
    
    filename=get_filename(category, False, log, what_to_plot)+"_evol"
    
    print("Animation for "+category+" =>",filename)

    data, days, arg_plot = prepare_data(begin=begin, end=end)
    data2, region_list, focus, title = prepare_graph(data, days, category=category)
    fcts = compute_fcts(data, sm)

    max_len = max( [ get_max_len( fcts[i][key] )  for i in range(len(fcts)) for key in fcts[i]  ] )
    for t in range(max_len):
        
        print("t=",t,"day=",days[t])

        # Crop all data 
        fcts2 = [ dict() for j in range(len(fcts)) ]
        for j in range(len(fcts)):
            for key in keys:
                fcts2[j][key] = dict()
                for f in data2[key]:
                    z = max( 0, t - max_len + len(fcts[j][key][f]) + 1 )
                    fcts2[j][key][f] = fcts[j][key][f][0:z]
                    
        fig = plot(region_list, what_to_plot, focus, fcts2, arg_plot, size=size, log=log, sync=sync)
        plt.suptitle(title, fontsize = 16)
        plt.savefig("tmp/"+filename+"_%02d.png"%t, dpi=dpi)
        plt.close('all')

    anim_command = "convert -verbose -delay 10 -loop 0 "
    lmax_len = list( range( 3,max_len-1 ) ) 
    src = " ".join([ "tmp/"+filename+"_%02d.png"%i for i in lmax_len ])
    print("Generate animations : "+src)
    #os.system( anim_command+src+" -delay 300 tmp/"+filename+"_%02d.png"%(max_len-1)+" ./fig/"+filename+".mp4" )
    os.system( anim_command+src+" -delay 300 tmp/"+filename+"_%02d.png"%(max_len-1)+" ./fig/"+filename+".gif" )


#########################
# Main global parameters



check_for_countries() 


dpi = 90 # graph quality (for png/mp4)


sm=10
sync=False

for var in [ ['deaths'], ['confirmed cases'] ]:
    for fcts in [ (1,2), (0,3), (0,1) ]:

        what_to_plot=[ var, fcts ]
        
        for (region, begin) in [
                ('World',''),
                ('top10',''),
                ('Europe','2/18/20'),
                ('Asia',''),
                ('North America','2/27/20'),
                ('South America','3/15/20'),
                ('Africa','3/9/20')#,
                #('Australia','')        
        ]:
            if fcts==(0,1) and not sync:
                animation(region, what_to_plot=what_to_plot, log=False, begin=begin, sm=sm)
            else: 
                curves(region, what_to_plot=what_to_plot, begin=begin, sm=sm, sync=sync)

