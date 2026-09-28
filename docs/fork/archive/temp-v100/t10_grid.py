import sqlite3, os

T = os.path.join(os.environ['TEMP'], 'v100')
for lbl in ['pp4096_d0', 'pp4096_d32k', 'pp32768']:
    con = sqlite3.connect(os.path.join(T, 't10_' + lbl + '.sqlite'))
    rows = con.execute(
        "select k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, count(*) "
        "from CUPTI_ACTIVITY_KIND_KERNEL k "
        "where (select value from StringIds where id = k.demangledName) like '%flash_attn%' "
        "group by 1,2,3,4,5,6").fetchall()
    print('===', lbl, '===')
    for gx, gy, gz, bx, by, bz, n in rows:
        print(f'  grid=({gx},{gy},{gz}) block=({bx},{by},{bz})  n={n}  -> CTAs={gx*gy*gz}, threads={bx*by*bz}')
    con.close()
