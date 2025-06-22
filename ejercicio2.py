import pandas as pd


df = pd.read_csv('synthetic_wtp_laptop_data.csv')


base_specs = {
    'Memory': 16,
    'Storage': 512,
    'CPU_class': 1,
    'Screen_size': 14.0,
}
base_price = 111000


upgrades = [
    {'name': 'Add 16 GB memory', 'Memory': 32, 'cost': 7000},
    {'name': 'Add 512 GB storage', 'Storage': 1024, 'cost': 5000},
    {'name': 'Upgrade CPU_class by 1 level', 'CPU_class': 2, 'cost': 15000},
    {'name': 'Increase screen size from 14 to 16 inches', 'Screen_size': 16.0, 'cost': 3000}
]


results = []
for upgrade in upgrades:
    specs = base_specs.copy()
    for k in upgrade:
        if k in specs:
            specs[k] = upgrade[k]
    # Filtrar el DataFrame para encontrar laptops con estas specs
    filtered = df
    for k, v in specs.items():
        filtered = filtered[filtered[k] == v]
    if not filtered.empty:
        market_price = filtered['price'].mean()
        gross_profit = market_price - (base_price + upgrade['cost'])
        results.append({'upgrade': upgrade['name'], 'gross_profit': gross_profit})


results = sorted(results, key=lambda x: x['gross_profit'], reverse=True)
for r in results[:2]:
    print(r['upgrade'], r['gross_profit'])