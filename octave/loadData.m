function D = loadData()

f = fopen("/Users/iain/code/personal/wine-quality/src/main/resources/wine.csv");
D = dlmread (f, ",");
fclose(f);

end
