# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:31:28 2022

@author: ereznic2
"""
# Specify file path to PyFAST
import sys
#sys.path.insert(1,'../PyFAST/')
# sys.path.remove(sys.path[5]) #ESG
sys.path.insert(1,sys.path[0] + '/ProFAST-main/')
import numpy as np
import pandas as pd
import ProFAST

# sys.path.append('../ProFAST/')

# import src.ProFAST as ProFAST

def run_profast_for_hydrogen(site_location,electrolyzer_size_mw,H2_Results,\
                            electrolyzer_system_capex_kw,time_between_replacement,electrolyzer_energy_kWh_per_kg,hydrogen_storage_capacity_kg,hydrogen_storage_cost_USDprkg,\
                            capex_desal,opex_desal,plant_life,water_cost,wind_size_mw,solar_size_mw,simple_cost_dict,revised_renewable_cost,wind_om_cost_kw,grid_connected_hopp,lcoe):
    
    # plant_life=useful_life
    # electrolyzer_system_capex_kw = electrolyzer_capex_kw
    
    # Estimate average efficiency and water consumption
    electrolyzer_efficiency_while_running = []
    water_consumption_while_running = []
    hydrogen_production_while_running = []
    for j in range(len(H2_Results['electrolyzer_total_efficiency'])):
        if H2_Results['hydrogen_hourly_production'][j] > 0:
            electrolyzer_efficiency_while_running.append(H2_Results['electrolyzer_total_efficiency'][j])
            water_consumption_while_running.append(H2_Results['water_hourly_usage'][j])
            hydrogen_production_while_running.append(H2_Results['hydrogen_hourly_production'][j])
    
    #electrolyzer_design_efficiency_HHV = np.max(electrolyzer_efficiency_while_running) # Should ideally be user input
    electrolyzer_average_efficiency_HHV = np.mean(electrolyzer_efficiency_while_running)
    water_consumption_avg_kgprhr = np.mean(water_consumption_while_running)
    
    water_consumption_avg_kgH2O_prkgH2 = water_consumption_avg_kgprhr/np.mean(hydrogen_production_while_running)
    
    water_consumption_avg_galH2O_prkgH2 = water_consumption_avg_kgH2O_prkgH2/3.79
    
    # Calculate average electricity consumption from average efficiency
    h2_HHV = 141.88
    elec_avg_consumption_kWhprkg = h2_HHV*1000/3600/electrolyzer_average_efficiency_HHV
    
    # Design point electricity consumption
    elec_consumption_kWhprkg_design = electrolyzer_energy_kWh_per_kg

    # Calculate electrolyzer production capacity
    #BELOW IS HOW IT WAS DONE (ESG)
    # electrolysis_plant_capacity_kgperday=   electrolyzer_size_mw/elec_consumption_kWhprkg_design*1000*24

    #electrolysis_plant_capacity_kgperday = electrolyzer_size_mw*electrolyzer_design_efficiency_HHV/h2_HHV*3600*24
    #H2_Results['H2 Aggregate'].loc['Stack Rated H2 Production [kg/hr]']
    #H2A_Results.loc['Total kWh/kg']
    #H2A_Results.loc['PEM Capacity Factor']
    #H2A_Results.loc['Avg [hrs] until Replacement Per Stack']
    electrolysis_plant_capacity_kgperday=H2_Results['H2 Aggregate'].loc['Stack Rated H2 Production [kg/hr]'].values[0]*electrolyzer_size_mw*24 #ESG
    # Installed capital cost
    electrolyzer_installation_factor = 12/100  #[%] for stack cost 
    
    # Indirect capital cost as a percentage of installed capital cost
    site_prep = 2/100   #[%]
    engineering_design = 10/100 #[%]
    project_contingency = 15/100 #[%]
    permitting = 15/100     #[%]
    land_cost = 250000   #[$]
    
    stack_replacement_cost = 15/100  #[% of installed capital cost]
    fixed_OM = 0.24     #[$/kg H2]    
    
    # Calculate electrolyzer installation cost
    total_direct_electrolyzer_cost_kw = (electrolyzer_system_capex_kw * (1+electrolyzer_installation_factor)) \
    
    electrolyzer_total_installed_capex = total_direct_electrolyzer_cost_kw*electrolyzer_size_mw*1000
    
    electrolyzer_indirect_cost = electrolyzer_total_installed_capex*(site_prep+engineering_design+project_contingency+permitting)
    
    #electrolyzer_installation_cost = electrolyzer_system_capex_kw*stack_installation_factor*electrolyzer_size_mw\
    #                               + electrolyzer_indirect_cost                             
    
    compressor_capex_USDprkWe_of_electrolysis = 39
    
    # Renewables system size
    # system_rating_mw = wind_size_mw + solar_size_mw #ESG commented out
     
    # Calculate capital costs
    capex_electrolyzer_overnight = electrolyzer_total_installed_capex + electrolyzer_indirect_cost
    capex_storage_installed = hydrogen_storage_capacity_kg*hydrogen_storage_cost_USDprkg
    capex_compressor_installed = compressor_capex_USDprkWe_of_electrolysis*electrolyzer_size_mw*1000
    #capex_hybrid_installed = hybrid_plant.grid.total_installed_cost
    # capex_hybrid_installed = revised_renewable_cost #ESG commented out
    # capex_wind_installed = wind_size_mw #ESG added

    # Fixed and variable costs
    fixed_OM = 12.8 #[$/kW-y]
    fixed_cost_electrolysis_total = fixed_OM*electrolyzer_size_mw*1000
    property_tax_insurance = 1.5/100    #[% of Cap/y]
    variable_OM = 1.30  #[$/MWh]
    
     # 
    if grid_connected_hopp == True:
        # If grid connected, conservatively assume electrolyzer runs with high CF
        # Or just take this straight from H2_Results if that works
        elec_cf = 0.97
    else:
        # If not grid connected, max DF will be relative to total renewable energy in
        elec_cf = H2_Results['cap_factor']

    # Amortized refurbishment expense [$/MWh]
    amortized_refurbish_cost = (total_direct_electrolyzer_cost_kw*stack_replacement_cost)\
            *max(((plant_life*8760*elec_cf)/time_between_replacement-1),0)/plant_life/8760/elec_cf*1000

    total_variable_OM = variable_OM+amortized_refurbish_cost
    
    total_variable_OM_perkg = total_variable_OM*elec_avg_consumption_kWhprkg/1000
    # solar_om_cost_kw = hybrid_plant.pv.om_capacity[0] #ESG added
    solar_om_cost_kw = simple_cost_dict['pv']['o&m_per_kw']
    wind_om_cost_kw = simple_cost_dict['wind']['o&m_per_kw']

    fixed_cost_wind = wind_om_cost_kw*simple_cost_dict['wind']['size_mw']*1000 #ESG added
    fixed_cost_solar = solar_om_cost_kw*simple_cost_dict['pv']['size_mw']*1000 #ESG added
    capex_wind_installed = simple_cost_dict['wind']['capex_per_kw'] * simple_cost_dict['wind']['size_mw']*1000
    capex_solar_installed = simple_cost_dict['pv']['capex_per_kw'] * simple_cost_dict['pv']['size_mw']*1000
    if simple_cost_dict['battery']['size_mw'] > 0:
        battery_hrs = simple_cost_dict['battery']['size_mwh']/simple_cost_dict['battery']['size_mw']
    else:
        battery_hrs=0
    battery_capex_per_kw= simple_cost_dict['battery']['capex_per_kwh']*battery_hrs +  simple_cost_dict['battery']['capex_per_kw']
    capex_battery_installed = battery_capex_per_kw * simple_cost_dict['battery']['size_mw']*1000
    fixed_cost_battery = simple_cost_dict['battery']['o&m_percent'] * capex_battery_installed
    # fixed_cost_renewables = wind_om_cost_kw*system_rating_mw*1000 #ESG commented out
    
    # Set up ProFAST
    pf = ProFAST.ProFAST('blank')
    
    # Fill these in - can have most of them as 0 also
    gen_inflation = 0.00
    pf.set_params('commodity',{"name":'Hydrogen',"unit":"kg","initial price":100,"escalation":gen_inflation})
    pf.set_params('capacity',electrolysis_plant_capacity_kgperday) #units/day
    pf.set_params('maintenance',{"value":0,"escalation":gen_inflation})
    pf.set_params('analysis start year',2022)
    pf.set_params('operating life',plant_life)
    pf.set_params('installation months',36)
    pf.set_params('installation cost',{"value":0,"depr type":"Straight line","depr period":4,"depreciable":False})
    pf.set_params('non depr assets',land_cost)
    pf.set_params('end of proj sale non depr assets',land_cost*(1+gen_inflation)**plant_life)
    pf.set_params('demand rampup',0)
    # pf.set_params('long term utilization',H2_Results['cap_factor'])
    pf.set_params('long term utilization',elec_cf)
    pf.set_params('credit card fees',0)
    pf.set_params('sales tax',0) 
    pf.set_params('license and permit',{'value':00,'escalation':gen_inflation})
    pf.set_params('rent',{'value':0,'escalation':gen_inflation})
    pf.set_params('property tax and insurance percent',property_tax_insurance)
    pf.set_params('admin expense percent',0)
    pf.set_params('total income tax rate',0.27)
    pf.set_params('capital gains tax rate',0.15)
    pf.set_params('sell undepreciated cap',True)
    pf.set_params('tax losses monetized',True)
    pf.set_params('operating incentives taxable',True)
    pf.set_params('general inflation rate',gen_inflation)
    pf.set_params('leverage after tax nominal discount rate',0.0824)
    pf.set_params('debt equity ratio of initial financing',1.38)
    pf.set_params('debt type','Revolving debt')
    pf.set_params('debt interest rate',0.0489)
    pf.set_params('cash onhand percent',1)
    
    #----------------------------------- Add capital items to ProFAST ----------------
    pf.add_capital_item(name="Electrolysis system",cost=capex_electrolyzer_overnight,depr_type="MACRS",depr_period=5,refurb=[0])
    pf.add_capital_item(name="Compression",cost=capex_compressor_installed,depr_type="MACRS",depr_period=5,refurb=[0])
    pf.add_capital_item(name="Hydrogen Storage",cost=capex_storage_installed,depr_type="MACRS",depr_period=5,refurb=[0])
    pf.add_capital_item(name ="Desalination",cost = capex_desal,depr_type="MACRS",depr_period=5,refurb=[0])
    # pf.add_capital_item(name = "Renewable Plant",cost = capex_hybrid_installed,depr_type = "MACRS",depr_period = 5,refurb = [0])
    pf.add_capital_item(name = "Wind Plant",cost = capex_wind_installed,depr_type = "MACRS",depr_period = 5,refurb = [0])
    pf.add_capital_item(name = "Solar Plant",cost = capex_solar_installed,depr_type = "MACRS",depr_period = 5,refurb = [0])
    pf.add_capital_item(name = "Battery Storage",cost = capex_battery_installed,depr_type = "MACRS",depr_period = 5,refurb = [0])
    #^ add solar plant like renewable plant, check other parts of code, have wind plant as separate
    total_capex = capex_electrolyzer_overnight+capex_compressor_installed+capex_storage_installed+capex_desal+ capex_wind_installed + capex_solar_installed + capex_battery_installed #ESG
    #capex_hybrid_installed

    capex_fraction = {'Electrolyzer':capex_electrolyzer_overnight/total_capex,
                      'Compression':capex_compressor_installed/total_capex,
                      'Hydrogen Storage':capex_storage_installed/total_capex,
                      'Desalination':capex_desal/total_capex,
                      'Wind Plant':capex_wind_installed/total_capex,
                      'Solar Plant':capex_solar_installed/total_capex,
                      'Battery Storage':capex_battery_installed/total_capex,
                      }
    
    #-------------------------------------- Add fixed costs--------------------------------
    pf.add_fixed_cost(name="Electrolyzer Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_electrolysis_total,escalation=gen_inflation)
    pf.add_fixed_cost(name="Desalination Fixed O&M Cost",usage=1.0,unit='$/year',cost=opex_desal,escalation=gen_inflation)
    # pf.add_fixed_cost(name="Renewable Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_renewables,escalation=gen_inflation)
    pf.add_fixed_cost(name="Wind Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_wind,escalation=gen_inflation)
    pf.add_fixed_cost(name="Solar Plant Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_solar,escalation=gen_inflation)
    pf.add_fixed_cost(name="Battery Storage Fixed O&M Cost",usage=1.0,unit='$/year',cost=fixed_cost_battery,escalation=gen_inflation)
    #^separate wind and solar here
    #---------------------- Add feedstocks, note the various cost options-------------------
    # pf.add_feedstock(name='Electricity',usage=elec_avg_consumption_kWhprkg,unit='kWh',cost=lcoe/100,escalation=gen_inflation)
    #ESG ADDED BELOW 03.09.2023
    # pf.add_feedstock(name='Electricity Cost',usage=elec_avg_consumption_kWhprkg,unit='$/kWh',cost=lcoe/100,escalation=gen_inflation)
    pf.add_feedstock(name='Electricity Cost',usage=elec_avg_consumption_kWhprkg,unit='kWh',cost=lcoe/100,escalation=gen_inflation)
    pf.add_feedstock(name='Water',usage=water_consumption_avg_galH2O_prkgH2,unit='gallon-water',cost=water_cost,escalation=gen_inflation)
    pf.add_feedstock(name='Var O&M',usage=1.0,unit='$/kg',cost=total_variable_OM_perkg,escalation=gen_inflation)
    #pf.add_feedstock(name='Grid Electricity Cost',usage= grid_electricity_usage,unit='$/kWh',cost=elec_price,escalation=gen_inflation)
        
    sol = pf.solve_price()
    
    summary = pf.summary_vals
    
    price_breakdown = pf.get_cost_breakdown()
    
    # Calculate financial expense associated with equipment
    cap_expense = price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]    
        
    # Calculate remaining financial expenses
    remaining_financial = price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
        + price_breakdown.loc[price_breakdown['Name']=='Property insurance','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]
    
    # Calculate LCOH breakdown and assign capital expense to equipment costs
    price_breakdown_electrolyzer = price_breakdown.loc[price_breakdown['Name']=='Electrolysis system','NPV'].tolist()[0] + cap_expense*capex_fraction['Electrolyzer']
    price_breakdown_compression = price_breakdown.loc[price_breakdown['Name']=='Compression','NPV'].tolist()[0] + cap_expense*capex_fraction['Compression']
    price_breakdown_storage = price_breakdown.loc[price_breakdown['Name']=='Hydrogen Storage','NPV'].tolist()[0]+cap_expense*capex_fraction['Hydrogen Storage']
    price_breakdown_desalination = price_breakdown.loc[price_breakdown['Name']=='Desalination','NPV'].tolist()[0] + cap_expense*capex_fraction['Desalination']
    # price_breakdown_renewables = price_breakdown.loc[price_breakdown['Name']=='Renewable Plant','NPV'].tolist()[0] + cap_expense*capex_fraction['Renewable Plant']
    price_breakdown_wind = price_breakdown.loc[price_breakdown['Name']=='Wind Plant','NPV'].tolist()[0] + cap_expense*capex_fraction['Wind Plant']
    price_breakdown_solar = price_breakdown.loc[price_breakdown['Name']=='Solar Plant','NPV'].tolist()[0] + cap_expense*capex_fraction['Solar Plant']
    price_breakdown_battery = price_breakdown.loc[price_breakdown['Name']=='Battery Storage','NPV'].tolist()[0] + cap_expense*capex_fraction['Battery Storage']

    price_breakdown_electrolysis_FOM = price_breakdown.loc[price_breakdown['Name']=='Electrolyzer Fixed O&M Cost','NPV'].tolist()[0]
    price_breakdown_electrolysis_VOM = price_breakdown.loc[price_breakdown['Name']=='Var O&M','NPV'].tolist()[0]
    price_breakdown_desalination_FOM = price_breakdown.loc[price_breakdown['Name']=='Desalination Fixed O&M Cost','NPV'].tolist()[0]
    # price_breakdown_renewables_FOM = price_breakdown.loc[price_breakdown['Name']=='Renewable Plant Fixed O&M Cost','NPV'].tolist()[0]  
    price_breakdown_wind_FOM = price_breakdown.loc[price_breakdown['Name']=='Wind Plant Fixed O&M Cost','NPV'].tolist()[0]  
    price_breakdown_solar_FOM = price_breakdown.loc[price_breakdown['Name']=='Solar Plant Fixed O&M Cost','NPV'].tolist()[0]  
    price_breakdown_battery_FOM = price_breakdown.loc[price_breakdown['Name']=='Battery Storage Fixed O&M Cost','NPV'].tolist()[0]    
    price_breakdown_taxes = price_breakdown.loc[price_breakdown['Name']=='Income taxes payable','NPV'].tolist()[0]\
        - price_breakdown.loc[price_breakdown['Name'] == 'Monetized tax losses','NPV'].tolist()[0]\
            
    if gen_inflation > 0:
        price_breakdown_taxes = price_breakdown_taxes + price_breakdown.loc[price_breakdown['Name']=='Capital gains taxes payable','NPV'].tolist()[0]

    price_breakdown_water = price_breakdown.loc[price_breakdown['Name']=='Water','NPV'].tolist()[0]
    #ESG ADDED BELOW 03.09.2023
    price_breakdown_elec_price = price_breakdown.loc[price_breakdown['Name']=='Electricity Cost','NPV'].tolist()[0]  
       
    # price_breakdown_financial = price_breakdown.loc[price_breakdown['Name']=='Non-depreciable assets','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Cash on hand reserve','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Property insurance','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Repayment of debt','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Interest expense','NPV'].tolist()[0]\
    #     + price_breakdown.loc[price_breakdown['Name']=='Dividends paid','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Sale of non-depreciable assets','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Cash on hand recovery','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Inflow of debt','NPV'].tolist()[0]\
    #     - price_breakdown.loc[price_breakdown['Name']=='Inflow of equity','NPV'].tolist()[0]
        #ESG commented out below
    # lcoh_check = price_breakdown_electrolyzer+price_breakdown_compression+price_breakdown_storage+price_breakdown_electrolysis_FOM\
    #     + price_breakdown_desalination+price_breakdown_desalination_FOM+ price_breakdown_electrolysis_VOM\
    #         +price_breakdown_renewables+price_breakdown_renewables_FOM+price_breakdown_taxes+price_breakdown_water+remaining_financial
    #ESG added below
    lcoh_check = price_breakdown_electrolyzer+price_breakdown_compression+price_breakdown_storage+price_breakdown_electrolysis_FOM\
        + price_breakdown_desalination+price_breakdown_desalination_FOM+ price_breakdown_electrolysis_VOM\
            +price_breakdown_wind + price_breakdown_solar + price_breakdown_battery+price_breakdown_wind_FOM\
                +price_breakdown_solar_FOM + price_breakdown_battery_FOM+price_breakdown_taxes+price_breakdown_water+remaining_financial\
                    +price_breakdown_elec_price
            
    lcoh_breakdown = {'LCOH: Compression & storage ($/kg)':price_breakdown_storage+price_breakdown_compression,\
                      'LCOH: Electrolyzer CAPEX ($/kg)':price_breakdown_electrolyzer,'LCOH: Desalination CAPEX ($/kg)':price_breakdown_desalination,\
                      'LCOH: Electrolyzer FOM ($/kg)':price_breakdown_electrolysis_FOM,'LCOH: Electrolyzer VOM ($/kg)':price_breakdown_electrolysis_VOM,\
                      'LCOH: Desalination FOM ($/kg)':price_breakdown_desalination_FOM,'LCOH: Wind Plant ($/kg)':price_breakdown_wind,\
                      'LCOH: Solar Plant ($/kg)': price_breakdown_solar,'LCOH: Battery Storage ($/kg)': price_breakdown_battery,\
                      'LCOH: Wind Plant FOM ($/kg)':price_breakdown_wind_FOM,'LCOH: Solar Plant FOM ($/kg)':price_breakdown_solar_FOM,\
                      'LCOH: Battery Storage FOM ($/kg)':price_breakdown_battery_FOM,'LCOH: Taxes ($/kg)':price_breakdown_taxes,\
                      'LCOH: Water consumption ($/kg)':price_breakdown_water,'LCOH: Electricity Cost ($/kg)':price_breakdown_elec_price,\
                      'LCOH: Finances ($/kg)':remaining_financial,'LCOH: total ($/kg)':lcoh_check}
    system_rating_mw = simple_cost_dict['wind']['size_mw'] + simple_cost_dict['pv']['size_mw'] #+ simple_cost_dict['battery']['size_mw']
    

    return(sol,summary,lcoh_breakdown,capex_electrolyzer_overnight/system_rating_mw/1000)