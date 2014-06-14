#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <ctime>
#include <cmath>

using namespace std;

double alpha_theta = 0.0001;
double alpha_x = 0.0001;
int lamda_theta = 100;
int lamda_x = 100;
const int numFeature = 200;
const int items = 102;
const int users = 25000;
const int iterations = 1000;

double y[items][users];
bool ir[items][users];
double arr_eh[items][users];
double predictRating[items][users];
double theta[users][numFeature];
double x[items][numFeature];
double meanY[items];

inline double h(int thetaj, int xi){
	double result = 0.0;
	for (int i = 0; i < numFeature; i++){
		result += theta[thetaj][i]*x[xi][i];
	}
	//cout << "h()" << endl;
	return result;
}
inline double eh(int thetaj, int xi){
	if (!ir[xi][thetaj]){
		return 0.0;
	}
	//cout << "eh()" << endl;
	return h(thetaj, xi)-y[xi][thetaj];
}
bool calAllIEh(int i){
	for (int j = 0; j < users; j++){
		arr_eh[i][j] = eh(j, i);
	}
	//cout << "calAllIEh end!" << endl;
	return true;
}
bool calAllJEh(int j){
	for (int i = 0; i < items; i++){
		arr_eh[i][j] = eh(j, i);
	}
	//cout << "calAllJEh end!" << endl;
	return true;
}
double shx(int k, int xi){
	double result = 0.0;
	for (int j = 0; j < users; j++){
		result += (arr_eh[xi][j]*theta[j][k]);
	}
	result += lamda_theta*x[xi][k];
	//cout << "shx()" << endl;
	return result;
}
double gdx(int k, int xi){
	double result;
	result = x[xi][k] - alpha_x*shx(k, xi);
	//cout << "gdx() " << endl;
	return result;
}
double shtheta(int k, int thetaj){
	double result = 0.0;
	for (int i = 0; i < items; i++){
		result += (arr_eh[i][thetaj]*x[i][k]);
	}
	result += lamda_theta*theta[thetaj][k];
	//cout << "shtheta() " << endl;
	return result;
}
double gdtheta(int k, int thetaj){
	double result;
	result = theta[thetaj][k] - alpha_theta*shtheta(k, thetaj);
	//cout << "gdtheta()" << endl;
	return result;
}


bool init(){
	memset(y, 0, sizeof(y));
	memset(ir, 0, sizeof(ir));
	srand((int)time(0));
	double r;
	int n;
	for (int i = 0; i < items; i++){
		for (int j = 0; j < numFeature; j++){
			n = rand();
			n = n % 20000;
			n -= 10000;
			r = n / (1e8);
			x[i][j] = r;
		}
		
	}
	for (int i = 0; i < users; i++){
		for (int j = 0; j < numFeature; j++){
			n = rand();
			n = n % 20000;
			n -= 10000;
			r = n / (1e8);
			theta[i][j] = r;
		}
	}
	cout << "init end!" << endl;
	return true;
}

bool norm(){
	for (int i = 0; i < items; i++){
		int count = 0;
		double tRating = 0.0;
		for (int j = 0; j < users; j++){
			if (!ir[i][j]) continue;
			count++;
			tRating += y[i][j];
		}
		meanY[i] = tRating / count;
	}
	for (int i = 0; i < items; i++){
		for (int j = 0; j < users; j++){
			if (!ir[i][j]) continue;
			y[i][j] -= meanY[i];
		}
	}
	cout << "norm end!" << endl;
	return true;
}

bool readData(string trFile){
	fstream tr(trFile, ios::in);
	bool isFirst = true;
	int id, userid, itemid;
	double rating;
	while (!tr.eof()){
		string temp = "";
		if (isFirst){
			isFirst = false;
			getline(tr, temp);
			continue;
		}
		tr >> id >> userid >> itemid >> rating;
		//cout << "userid: " << userid << " itemid: " << itemid << "rating: " << rating << endl;
		y[itemid][userid] = rating;
		ir[itemid][userid] = true;
		//cout << "y["<<itemid<<"]["<<userid<<"]: " << y[itemid][userid] << endl;
		getline(tr, temp);
	}
	tr.close();
	
	norm();
	//for (int i = 0; i < users; i++){
	//	for (int j = 0; j < items; j++){
	//		/*if (abs(y[j][i]) < 1e-6){
	//		cout << "y[j][i] ====== 0" << endl;
	//		}*/
	//		cout << "y[" << i << "][" << j << "]: " << y[j][i] << endl;
	//	}
	//}
	cout << "read data end!" << endl;
	return true;
}
bool writeData(string testFile, string resultFile){
	fstream test(testFile, ios::in);
	fstream result(resultFile, ios::out);
	bool isFirst = true;
	int id, userid, itemid;
	while (!test.eof()){
		string temp = "";
		if (isFirst){
			isFirst = false;
			getline(test, temp);
			result << "id,rating\n";
			continue;
		}
		test >> id >> userid >> itemid;
		result << id << "," << h(userid, itemid)+meanY[itemid] << "\n";
	}
	test.close();
	result.close();
	cout << "write data end!" << endl;
	return true;
}
bool cf(){

	for (int j = 0; j < users; j++){
		calAllJEh(j);
		for (int k = 0; k < numFeature; k++){
			//cout << "feature: " << k << ", user: " << j << endl;
			theta[j][k] = gdtheta(k, j);
		}
	}
	
	for (int i = 0; i < items; i++){
		calAllIEh(i);
		for (int k = 0; k < numFeature; k++){
			//cout << "feature: " << k << ", item: " << i << endl;
			x[i][k] = gdx(k, i);
		}
	}

	cout << "cf end!" << endl;
	return true;
}
int main(){
	init();
	readData("train.txt");
	for (int i = 0; i < iterations; i++){
		cout << "pass " << i << endl;
		cf();
	}
	writeData("test.txt", "linresult.csv");
}