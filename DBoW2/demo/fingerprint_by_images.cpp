/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <opencv2/ximgproc.hpp>



#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<cv::Mat > > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void fingerprint_by_images(vector<vector<cv::Mat > > &features);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 1262;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// ----------------------------------------------------------------------------

void convertir (const char * origen)
{
        char comando[256];
//        sprintf(comando, "convert %s -gravity Center -crop 60%cx100%c+0+0 +repage /tmp/output2.png", origen, '%', '%');

 //      system(comando);

       // sprintf(comando, "convert %s   -gravity North -region 100%cx50%c -fill black -colorize 100%c /tmp/output.png", origen, '%', '%', '%');
       //system(comando);

        sprintf(comando, "convert %s -gravity South -crop 100%cx50%c+0+0 +repage /tmp/m/output.png  ", origen, '%', '%');
       system(comando);

}


// ----------------------------------------------------------------------------

int main()
{
  vector<vector<cv::Mat > > features;
  loadFeatures(features);

  fingerprint_by_images(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features)
{
	features.clear();
	features.reserve(NIMAGES);

  //cv::Ptr<cv::ORB> orb = cv::ORB::create(8000);
  // ULTIMO MUY BIEN cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 15, 85, 2, 4, cv::ORB::HARRIS_SCORE, 75);
  //cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 15, 85, 2, 4, cv::ORB::HARRIS_SCORE, 75);
  // cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 9, 55, 2, 2, cv::ORB::HARRIS_SCORE, 85);
  // BIEN cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 5, 90, 1, 2, cv::ORB::HARRIS_SCORE, 30);
  // EL MEJOR! BUEN PERFORMANCE cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 3, 65, 2, 4, cv::ORB::HARRIS_SCORE, 45);
  cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 3, 65, 2, 4, cv::ORB::HARRIS_SCORE, 45);
	// cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 3, 65, 2, 4, cv::ORB::HARRIS_SCORE, 45);

	cout << "Extracting ORB features..." << endl;
	for(int i = 0; i < 79; ++i) {
		stringstream ss;
		// ss << "f" << i << ".jpg";
		ss << "f0.jpg";

		cv::Mat image = cv::imread(ss.str(), cv::IMREAD_COLOR);

		cv::Mat mask;
		vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;

		orb->detectAndCompute(image, mask, keypoints, descriptors);
		features.push_back(vector<cv::Mat >());
		changeStructure(descriptors, features.back());
	}

	/*
	for(int i = 80; i < NIMAGES; ++i) {
		stringstream ss;
		ss << "f" << i << ".jpg";

		cv::Mat image = cv::imread(ss.str(), cv::IMREAD_COLOR);

		cv::Mat mask;
		vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;

		orb->detectAndCompute(image, mask, keypoints, descriptors);

		// Dibujar los puntos clave en la imagen
		// cv::Mat image_with_keypoints;
		// drawKeypoints(image, keypoints, image_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

		// Mostrar la imagen con los puntos clave detectados
		// cv::namedWindow("ORB Keypoints", cv::WINDOW_NORMAL);
		// cv::imshow("ORB Keypoints", image_with_keypoints);

		// Esperar a que se presione una tecla para cerrar la ventana
		// cv::waitKey(0);

		features.push_back(vector<cv::Mat >());
		changeStructure(descriptors, features.back());
	}
	*/
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------




std::vector<BowVector> loadBowVectors(const std::string& filename) {
	std::ifstream ifs(filename, std::ios::binary);
	std::vector<BowVector> bowVectors;

	while (ifs.peek() != EOF) {
		BowVector bv;
		size_t size;
		ifs.read(reinterpret_cast<char*>(&size), sizeof(size)); // Leer el tamaño del BowVector
		for (size_t i = 0; i < size; ++i) {
			WordId id;
			WordValue value;
			ifs.read(reinterpret_cast<char*>(&id), sizeof(id));     // Leer WordId
			ifs.read(reinterpret_cast<char*>(&value), sizeof(value)); // Leer WordValue
			bv[id] = value;
		}
		bowVectors.push_back(bv);
	}
	ifs.close();
	return bowVectors;
}


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

void fingerprint_by_images(vector<vector<cv::Mat > > &features)
{
	// branching factor and depth levels 
	// const int k = 9;
	// const int L = 3;
	// const int k = 15;
	const int k = 50;
	const int L = 3;
	const WeightingType weight = TF_IDF;
	const ScoringType scoring = L1_NORM;

	cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
	OrbVocabulary voc(k, L, weight, scoring);
	// voc.create(features);
	//OrbVocabulary voc;
	voc.load("db_vocabulary1.yml.gz");  // Cargar el vocabulario guardado

	cout << "... done!" << endl;

	cout << "Vocabulary information: " << endl
	<< voc << endl << endl;

	// lets do something with this vocabulary
	cout << "Tratando de localizar cada imagen en la BD: " << endl;

	// Cargar los BowVectors desde un archivo
	std::vector<BowVector> bowVectors = loadBowVectors("db_bowvectors1.dat");


	// otra vez usaremos orb
	cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 3, 65, 2, 4, cv::ORB::HARRIS_SCORE, 45);



	// vemos si podemos encontrar el arbol

  	int i, j;
	int arbol = 0;
	int arbol_prev = -1;
	int varios = 0;
  	double score; double ac; double total; int kk;
  	BowVector v1, v2;
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	for (i=80; i<1200; i++) {

		// Captura el tiempo final
		auto end = std::chrono::high_resolution_clock::now();
		// Calcula la duración
		std::chrono::duration<double> duration = end - start;
		// Imprime la duración en segundos
		std::cout << "Tiempo transcurrido foto : " << i-1 << "  " << duration.count() << " segundos" << std::endl;
		// Inicia un nuevo cronometro
		start = std::chrono::high_resolution_clock::now();


		// cargar image y obtener puntos claves y descriptores

		stringstream ss;
		ss << "f" << i << ".jpg";

		cv::Mat image = cv::imread(ss.str(), cv::IMREAD_COLOR);

		cv::Mat mask;
		vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;

		orb->detectAndCompute(image, mask, keypoints, descriptors);

		features.push_back(vector<cv::Mat >());
		changeStructure(descriptors, features.back());

		



		ac=0; kk=0; total=0;
		for (j=0; j<79; j++) {
    			//voc.transform(features[i], v1);
    			voc.transform(features.back(), v1);
			score = voc.score(v1, bowVectors[j]);
			//cout << "score " << i << " " << j << "  " << score << endl; 
			ac = ac + score;
			kk++;
			if (kk == 5) {
				if (ac > total) {
					total = ac;
					arbol = j;
				}
				kk=0; 
				ac=0;
			}
		}
		if (arbol == arbol_prev) {
			varios++;
			// SI hay al menos 4 coincidencias consecutivas
			if (varios>=5) {
  				cout << " arbol " << i << " coincide con " << arbol << endl;
			};
		} else {
			varios = 0;
		}
		arbol_prev = arbol;
	}


}

// ----------------------------------------------------------------------------
