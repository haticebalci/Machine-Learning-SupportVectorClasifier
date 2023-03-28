# Machine-Learning-SupportVectorClasifier
# SupportVectorClasifier
Bu Python kodu, Destek Vektör Sınıflandırıcı (SVC) makine öğrenimi algoritmasını kullanarak İris veri kümesini sınıflandırmak için yazılmıştır. Kodun kısaca açıklaması aşağıdaki gibidir:

- Gerekli kütüphaneler, pandas, numpy ve matplotlib gibi kütüphaneler dahil olmak üzere içe aktarılır.
- İris veri kümesi, pandas kütüphanesinin read_excel yöntemi kullanılarak içe aktarılır.
- Veri kümesi, scikit-learn kütüphanesinin train_test_split yöntemi kullanılarak eğitim ve test setlerine ayrılır.
- Veriler, Standard Scaler adlı bir ön işleme tekniği kullanılarak önceden işlenir, bu işlem verileri sıfır ortalama ve birim varyansa sahip olacak şekilde ölçeklendirir.
- SVC algoritması, eğitim verileri üzerinde modeli eğitir ve test verilerinde tahminler yapar.
- Modelin doğruluğu, bir karışıklık matrisi kullanılarak değerlendirilir.
- Genel olarak, kod SVC algoritmasını kullanarak basit bir sınıflandırma problemi uygular ve verileri bölmeyi, önceden işlemeyi ve modelin doğruluğunu değerlendirmeyi gösterir.
