
python EDA13.py

echo "base feature is ok"

python EDA16-fourWeek.py
python EDA16-fourWeek_rightTime.py
python EDA16-threeWeek.py
python EDA16-threeWeek_rightTime.py
python EDA16-twoWeek.py

echo "A result is ok"

python sbb_train1.py
python sbb_train2.py
python sbb_train3.py
echo "B win size 3 is ok"
python sbb2_train1.py
python sbb2_train2.py
python sbb2_train3.py
echo "B win size 2 is ok"
python sbb4_train1.py
python sbb4_train2.py
python sbb4_train3.py
echo "B win size 4 is ok"
python gen_result.py

echo "finish,,,,,"
