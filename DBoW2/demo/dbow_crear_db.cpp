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
void saveBowVectors(const vector<vector<cv::Mat > > &features, const std::string& filename);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 79;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}


// ----------------------------------------------------------------------------

void convertir (const char * origen)
{
        char comando[256];
	// sprintf(comando, "convert %s -gravity Center -crop 60%cx100%c+0+0 +repage /tmp/output2.png", origen, '%', '%');
	// system(comando);

	// sprintf(comando, "convert %s   -gravity North -region 100%cx50%c -fill black -colorize 100%c /tmp/output.png", origen, '%', '%', '%');
	// system(comando);

	sprintf(comando, "convert %s -gravity South -crop 100%cx50%c+0+0 +repage /tmp/m/output.png  ", origen, '%', '%');
	system(comando);

}


// ----------------------------------------------------------------------------

int main()
{
  vector<vector<cv::Mat > > features;
  loadFeatures(features);

  saveBowVectors(features, "db_bowvectors1.dat");

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features)
{
	features.clear();
	features.reserve(NIMAGES);

  //cv::Ptr<cv::ORB> orb = cv::ORB::create(8000);
  // cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 9, 55, 2, 2, cv::ORB::HARRIS_SCORE, 85);
  // BIEN cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 5, 90, 1, 2, cv::ORB::HARRIS_SCORE, 30);
  // EL MEJOR! BUEN PERFORMANCE cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 3, 65, 2, 4, cv::ORB::HARRIS_SCORE, 45);
  cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 3, 65, 2, 4, cv::ORB::HARRIS_SCORE, 45);
  // ULTIMO MUY BIEN cv::Ptr<cv::ORB> orb = cv::ORB::create(200, 1.01, 15, 85, 2, 4, cv::ORB::HARRIS_SCORE, 75);

	// montamos el ram fs
	// system("mkdir /tmp/m");
	// printf("clave de root:\n"); fflush(0);
	// system("sudo mount -t tmpfs ramfs /tmp/m ");

	cout << "Extracting ORB features..." << endl;

	for(int i = 0; i < NIMAGES; ++i) {
		stringstream ss;
		ss << "f" << i << ".jpg";

		//cv::Mat image = cv::imread(ss.str(), 0);
		cv::Mat image = cv::imread(ss.str(), cv::IMREAD_COLOR);

		// si hay que convertir
		// std::string str = ss.str();
		// convertir(str.c_str());
		// cv::Mat image = cv::imread("/tmp/m/output.png", cv::IMREAD_COLOR);

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
		//  cv::waitKey(0);


		features.push_back(vector<cv::Mat >());
		changeStructure(descriptors, features.back());
	}
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



void saveBowVectors(const vector<vector<cv::Mat > > &features, const std::string& filename) {
	// branching factor and depth levels 
	//const int k = 9;
	//const int L = 3;
	// const int k = 15;
	const int k = 50;
	const int L = 3;
	const WeightingType weight = TF_IDF;
	const ScoringType scoring = L1_NORM;

	OrbVocabulary voc(k, L, weight, scoring);

	cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
	voc.create(features);
	cout << "... done!" << endl;

	cout << "Vocabulary information: " << endl
	<< voc << endl << endl;

	std::vector<BowVector> bowVectors(NIMAGES);

	// Precomputar los BowVector
	for (int i = 0; i < NIMAGES; i++) {
    		voc.transform(features[i], bowVectors[i]);
	}

	// Guardar vocabulario
	voc.save("db_vocabulary1.yml.gz");  // Guardar en un archivo comprimido


	// Guardar los BowVectors en un archivo
	//saveBowVectors(bowVectors, "bowvectors1.dat");
	
	std::ofstream ofs(filename, std::ios::binary);
	for (const auto& bv : bowVectors) {
		size_t size = bv.size();
		ofs.write(reinterpret_cast<const char*>(&size), sizeof(size)); // Escribir el tamaño del BowVector
		for (const auto& pair : bv) {
			ofs.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));  // Escribir WordId
			ofs.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second)); // Escribir WordValue
        	}
	}
	ofs.close();
}


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



