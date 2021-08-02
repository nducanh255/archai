python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,2,2,8,8 --d_model 64 --d_head 16,32,32,8,8 --d_inner 710,1021,1683,1053,1748 --d_embed 512 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,2,4,4,2,8,8,4 --d_model 128 --d_head 64,64,32,32,64,16,16,32 --d_inner 1451,637,1959,745,1354,1319,1855,1490 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,8,4,2,2,8 --d_model 128 --d_head 16,16,32,64,64,16 --d_inner 1873,1904,1921,1322,1086,1492 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,4,4,8,4,2 --d_model 128 --d_head 32,32,32,16,32,64 --d_inner 662,2021,1786,1994,1215,1607 --d_embed 256 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,4,8,8,8 --d_model 64 --d_head 16,16,8,8,8 --d_inner 993,1418,1796,933,1264 --d_embed 512 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,8,8,2,2,2,2 --d_model 64 --d_head 16,8,8,32,32,32,32 --d_inner 1010,1629,2004,1286,734,592,886 --d_embed 512 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,4,8,4,2 --d_model 64 --d_head 16,16,8,16,32 --d_inner 849,528,1319,1605,1834 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,2,8,8,2 --d_model 512 --d_head 256,256,64,64,256 --d_inner 1250,1865,1294,1983,1723 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,2,4,4,4,8,8 --d_model 64 --d_head 8,32,16,16,16,8,8 --d_inner 1344,785,1812,1952,559,616,1413 --d_embed 256 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,8,8,2,2 --d_model 512 --d_head 256,64,64,256,256 --d_inner 1812,1410,1467,1050,2019 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,2,8,8,2,8,8,2 --d_model 512 --d_head 256,256,64,64,256,64,64,256 --d_inner 1326,1613,1752,1318,1958,1938,2012,1352 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,8,4,4,8,4,2 --d_model 64 --d_head 32,8,16,16,8,16,32 --d_inner 550,1976,1281,1685,1719,750,788 --d_embed 512 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,4,8,4,2 --d_model 128 --d_head 32,32,16,32,64 --d_inner 780,1823,730,1136,727 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,4,2,2,4,8,4 --d_model 512 --d_head 256,128,256,256,128,64,128 --d_inner 1248,1780,2042,1643,1800,1326,1167 --d_embed 256 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,8,2,8,2 --d_model 64 --d_head 8,8,32,8,32 --d_inner 985,1242,1892,726,1574 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,2,8,8,2,4 --d_model 256 --d_head 128,128,128,32,32,128,64 --d_inner 1396,1609,882,657,1675,1040,632 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,4,4,2,8,2,8 --d_model 64 --d_head 16,16,16,32,8,32,8 --d_inner 1832,1822,820,1585,1947,704,1195 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,8,2,8,4,4,2 --d_model 512 --d_head 64,64,256,64,128,128,256 --d_inner 1480,1378,1727,1472,1545,1610,1691 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,8,4,2,2 --d_model 64 --d_head 8,8,16,32,32 --d_inner 1979,1489,1725,609,827 --d_embed 256 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,4,2,4,4,4 --d_model 64 --d_head 16,16,32,16,16,16 --d_inner 1037,1516,1339,1539,1210,1154 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,8,8,8,2,4,4 --d_model 512 --d_head 256,64,64,64,256,128,128 --d_inner 1712,1715,1801,1069,1990,1449,1815 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,2,2,4,4 --d_model 64 --d_head 32,8,32,32,16,16 --d_inner 1508,794,1657,630,859,1892 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 8,8,4,2,8,2,8,2 --d_model 512 --d_head 64,64,128,256,64,256,64,256 --d_inner 1028,1586,1086,1085,1786,1315,1980,1128 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,4,8,2,4,2,8 --d_model 128 --d_head 32,32,16,64,32,64,16 --d_inner 1016,1203,1510,1261,1998,1965,693 --d_embed 512 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,2,2,2,4 --d_model 256 --d_head 128,128,128,128,64 --d_inner 1299,952,937,1814,879 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,4,4,4,2,8,8 --d_model 512 --d_head 256,128,128,128,256,64,64 --d_inner 1794,1698,1313,1191,1545,1146,1689 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,4,8,4,8,4 --d_model 64 --d_head 32,32,16,8,16,8,16 --d_inner 1694,869,1820,1271,1139,1369,1368 --d_embed 256 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,8,8,8,8 --d_model 64 --d_head 32,8,8,8,8 --d_inner 1916,612,1051,787,1216 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,8,4,8,8 --d_model 128 --d_head 64,16,32,16,16 --d_inner 1384,1113,1977,594,690 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,4,4,2,2,8 --d_model 512 --d_head 64,128,128,256,256,64 --d_inner 2005,1470,1328,1283,1562,1294 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,4,2,8,2 --d_model 256 --d_head 32,64,128,32,128 --d_inner 1488,1577,1223,1395,873 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,4,8,2,2,2 --d_model 512 --d_head 128,128,64,256,256,256 --d_inner 1310,1789,1676,1579,1034,1132 --d_embed 512 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,8,2,4,8 --d_model 512 --d_head 256,64,64,256,128,64 --d_inner 1709,1541,1311,1361,1860,1527 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,4,8,8,8,2,2,2 --d_model 512 --d_head 128,128,64,64,64,256,256,256 --d_inner 1986,1801,1916,1195,1771,1907,1939,1447 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,4,4,8,8,2,2 --d_model 512 --d_head 128,128,128,64,64,256,256 --d_inner 1351,1367,1445,1823,1590,1897,1275 --d_embed 512 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,2,4,2,8 --d_model 512 --d_head 64,256,128,256,64 --d_inner 1888,1374,1073,2012,1475 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,2,8,2,4 --d_model 512 --d_head 256,256,64,256,128 --d_inner 1686,1300,1985,1489,1931 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,2,8,8,4 --d_model 256 --d_head 32,128,32,32,64 --d_inner 1321,1095,1603,1183,1896 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,4,4,4,2,8,8,2 --d_model 128 --d_head 32,32,32,32,64,16,16,64 --d_inner 1399,1221,1516,1625,989,626,847,1434 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,4,2,4,2,8 --d_model 128 --d_head 16,32,64,32,64,16 --d_inner 799,1676,1303,734,1556,1306 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,4,4,8,8,8 --d_model 512 --d_head 64,128,128,64,64,64 --d_inner 2012,1999,1713,1484,1236,1937 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,8,2,8,4,8 --d_model 64 --d_head 8,8,32,8,16,8 --d_inner 1858,757,968,1782,1769,653 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,2,8,2,2,2 --d_model 512 --d_head 64,256,64,256,256,256 --d_inner 1552,1283,1857,1844,1046,2047 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,2,4,8,2,4,2 --d_model 512 --d_head 128,256,128,64,256,128,256 --d_inner 1588,1048,1880,1504,1837,1024,1264 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,2,2,4,4,8,2 --d_model 256 --d_head 32,128,128,64,64,32,128 --d_inner 882,1163,600,1253,1454,1490,1153 --d_embed 512 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,8,4,2,2 --d_model 128 --d_head 16,16,32,64,64 --d_inner 1700,1313,1395,1350,1644 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,2,2,8,4,2 --d_model 512 --d_head 256,256,256,64,128,256 --d_inner 1962,1218,1602,1347,1126,1238 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,2,2,4,2,2,2,4 --d_model 64 --d_head 16,32,32,16,32,32,32,16 --d_inner 922,1526,1568,1958,526,1496,1221,995 --d_embed 512 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,2,4,2,4,4 --d_model 128 --d_head 16,64,32,64,32,32 --d_inner 544,810,1538,1656,1244,1020 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,2,2,4,4,4,2,4 --d_model 64 --d_head 16,32,32,16,16,16,32,16 --d_inner 1353,1836,1411,1116,1669,1053,1367,892 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 8,2,8,8,4,8,4,4 --d_model 64 --d_head 8,32,8,8,16,8,16,16 --d_inner 1016,1795,1529,1868,711,1584,727,655 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,8,8,8,8,4 --d_model 64 --d_head 8,8,8,8,8,16 --d_inner 1256,1503,537,1292,1990,634 --d_embed 512 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,4,8,8,4,8 --d_model 256 --d_head 64,64,32,32,64,32 --d_inner 1760,529,1261,1226,810,1734 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,2,4,2,4 --d_model 512 --d_head 64,256,128,256,128 --d_inner 1338,1367,1113,1353,2005 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,8,8,2,4 --d_model 128 --d_head 64,16,16,16,64,32 --d_inner 882,1995,1058,1792,1912,526 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,8,2,2,4,2,4 --d_model 128 --d_head 64,16,64,64,32,64,32 --d_inner 1461,570,1038,1931,1564,676,1110 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,4,4,8,2,2 --d_model 128 --d_head 64,64,32,32,16,64,64 --d_inner 1307,1082,1231,835,831,802,1525 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,4,4,4,4,4 --d_model 256 --d_head 128,128,64,64,64,64,64 --d_inner 717,1098,1821,911,1770,2039,1606 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,4,2,4,8 --d_model 256 --d_head 128,64,128,64,32 --d_inner 1656,1274,702,1221,1154 --d_embed 512 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,8,4,8,4 --d_model 256 --d_head 32,32,64,32,64 --d_inner 1501,1482,1967,745,1563 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,4,4,2,4,4,8,8 --d_model 256 --d_head 64,64,64,128,64,64,32,32 --d_inner 1961,1595,1196,1204,1515,549,899,1977 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,8,2,2,4 --d_model 128 --d_head 64,16,16,64,64,32 --d_inner 1777,614,1458,801,1365,650 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,4,4,8,4,4,2,2 --d_model 256 --d_head 128,64,64,32,64,64,128,128 --d_inner 1999,1558,699,1579,1478,1257,1601,649 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,2,4,4,2,2 --d_model 128 --d_head 16,64,32,32,64,64 --d_inner 1139,566,540,874,929,934 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,2,2,8,2 --d_model 512 --d_head 128,256,256,64,256 --d_inner 1588,1141,1913,1071,1514 --d_embed 512 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 8,2,8,8,2 --d_model 128 --d_head 16,64,16,16,64 --d_inner 1829,1880,1034,1063,581 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,4,2,4,2 --d_model 128 --d_head 64,32,64,32,64 --d_inner 1023,1251,1881,589,778 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,2,4,8,2 --d_model 256 --d_head 128,32,128,64,32,128 --d_inner 651,1148,1924,802,1489,1679 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,2,8,2,8,8,4,4 --d_model 64 --d_head 32,32,8,32,8,8,16,16 --d_inner 1268,547,556,692,1191,1853,1324,1857 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 8,2,2,4,8,8,4,8 --d_model 256 --d_head 32,128,128,64,32,32,64,32 --d_inner 1345,1308,1097,1100,1170,1967,2048,1237 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,8,8,8,4,2 --d_model 64 --d_head 16,8,8,8,16,32 --d_inner 680,1191,1074,1072,1123,858 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,4,8,4,4,2,4 --d_model 64 --d_head 32,16,8,16,16,32,16 --d_inner 999,847,951,514,1863,1736,1031 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,8,4,4,2,4 --d_model 256 --d_head 64,32,64,64,128,64 --d_inner 1213,1576,1625,666,1684,1277 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,8,2,8,2,8 --d_model 128 --d_head 64,64,16,64,16,64,16 --d_inner 923,1885,1861,1171,930,886,1604 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,8,8,4,2,2 --d_model 256 --d_head 64,32,32,64,128,128 --d_inner 551,1416,1599,1507,1105,1726 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,2,8,8,2,4,2,2 --d_model 256 --d_head 128,128,32,32,128,64,128,128 --d_inner 1027,1445,1755,1478,645,1436,1132,513 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 2,2,2,8,8 --d_model 512 --d_head 256,256,256,64,64 --d_inner 1231,2000,1170,1185,1785 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,4,4,8,2,4,2,8 --d_model 512 --d_head 256,128,128,64,256,128,256,64 --d_inner 1305,1454,1687,1943,1768,1305,1370,2029 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,2,4,4,4,4,4 --d_model 512 --d_head 128,256,128,128,128,128,128 --d_inner 1051,1947,1989,1935,1211,1186,1879 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,8,4,2,2,4,4,8 --d_model 512 --d_head 256,64,128,256,256,128,128,64 --d_inner 1905,1455,1755,1939,1467,1318,1526,1527 --d_embed 256 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 4,8,8,2,8,4 --d_model 64 --d_head 16,8,8,32,8,16 --d_inner 1930,1974,1209,619,1903,571 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,4,2,2,4,4 --d_model 128 --d_head 64,32,64,64,32,32 --d_inner 1229,1618,1901,952,1496,987 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,4,2,4,8,4 --d_model 128 --d_head 16,32,64,32,16,32 --d_inner 1152,1119,1991,1163,1652,1064 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,2,8,2,8,2,2 --d_model 128 --d_head 16,64,16,64,16,64,64 --d_inner 635,1880,1112,1226,1812,1517,1135 --d_embed 512 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,2,8,4,8,4,8,8 --d_model 256 --d_head 64,128,32,64,32,64,32,32 --d_inner 537,1332,1889,1375,2042,1368,1218,559 --d_embed 256 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 5 --n_head 4,8,4,4,8 --d_model 512 --d_head 128,64,128,128,64 --d_inner 1178,1837,1311,1401,1564 --d_embed 128 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 2,2,8,8,2,4,2 --d_model 512 --d_head 256,256,64,64,256,128,256 --d_inner 1034,1942,1141,1043,1923,1334,1596 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,2,2,2,4,8,8 --d_model 128 --d_head 16,64,64,64,32,16,16 --d_inner 839,827,588,1771,664,864,1055 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 8,8,4,2,2,2,2,4 --d_model 128 --d_head 16,16,32,64,64,64,64,32 --d_inner 1949,1236,1618,1507,733,1609,883,1647 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,2,8,4,2,2,8,8 --d_model 128 --d_head 64,64,16,32,64,64,16,16 --d_inner 861,1569,1010,1574,1867,1201,927,1235 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,4,4,4,2 --d_model 512 --d_head 256,64,128,128,128,256 --d_inner 1171,1419,1177,1368,1833,1439 --d_embed 128 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 8,8,8,2,4,4,8 --d_model 128 --d_head 16,16,16,64,32,32,16 --d_inner 1066,1366,1545,593,1941,671,1698 --d_embed 512 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 4,2,2,4,8,8,8,8 --d_model 256 --d_head 64,128,128,64,32,32,32,32 --d_inner 1387,670,1488,955,558,2039,1794,1334 --d_embed 512 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,2,4,8,2 --d_model 64 --d_head 32,8,32,16,8,32 --d_inner 1017,1927,1892,1055,1577,1074 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 7 --n_head 4,4,4,2,8,4,8 --d_model 64 --d_head 16,16,16,32,8,16,8 --d_inner 911,773,1356,542,1240,548,651 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 2,8,4,8,4,2 --d_model 256 --d_head 128,32,64,32,64,128 --d_inner 1585,1916,969,849,1939,1097 --d_embed 256 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 2,2,8,8,2,8,2,4 --d_model 128 --d_head 64,64,16,16,64,16,64,32 --d_inner 1583,1637,721,556,1773,1104,1817,1719 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 8 --n_head 8,8,8,2,2,4,8,2 --d_model 256 --d_head 32,32,32,128,128,64,32,128 --d_inner 1128,1377,1989,783,1087,670,2042,1308 --d_embed 512 --div_val 2 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,2,8,4,2,4 --d_model 128 --d_head 16,64,16,32,64,32 --d_inner 773,1724,935,1631,1699,536 --d_embed 128 --div_val 1 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
python archai/nlp/nvidia_transformer_xl/mem_transformer.py --n_layer 6 --n_head 8,2,2,4,8,8 --d_model 256 --d_head 32,128,128,64,32,32 --d_inner 1802,1561,1234,1713,514,1791 --d_embed 256 --div_val 4 
if [ $? -ne 0 ]; then 
 echo FAIL 
 exit 
 fi 
