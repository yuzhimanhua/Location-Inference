#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
//#include <windows.h>
#include <math.h>
#include <random>
#include <string>
using namespace std;

#define MAXD 350000
#define MAXT 5
#define MAXV 30000
#define MAXL 40
#define MAXN 140
#define MAXF 5
#define MAXC 150

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(0,1);

int D, //Number of Training Documents
    D0, //Number of Training + Testing Documents
    T = 1, //Number of Topics
    V = 28500, //Size of Vocabulary
    L = 30, //Number of Locations
    F = 4, //Number of Features
    maxIter = 100,
    burnIn = 20,
    EMStep = 5,
    paraStep = 20,

    //Name of Variables n: Document-Word-Topic-Location
    kxxx[MAXD] = {0}, //Number of words in Document d
    kxjx[MAXD][MAXT] = {0},
    xrji[MAXV][MAXT][MAXL] = {0},
    xxji[MAXT][MAXL] = {0},
    xxxi[MAXL] = {0},

    //Name of Variables m: Document-Feature-Location-Class
    xuiv[MAXF][MAXL][MAXC] = {0},
    xuix[MAXF][MAXL] = {0},

    //Topics and words
    z[MAXD][MAXN] = {0}, //Topic assignments for each word
    w[MAXD][MAXN] = {0}, //All documents
    p[MAXD] = {0}, //Location assignments for each document
    x[MAXD][MAXF] = {0}, //All categorical features
    Cl[MAXF] = {99, 35, 43, 104, 24}, //Number of the categories that each feature has
    num[MAXL] = {0};

float alpha = 0, beta = 0.01, gamma_ = 0.01, delta = 0.01, //Hyper-Parameters
      mu[MAXL][2] = {0}, //Center of each region
      sigma2[MAXL] = {0}, //Variance of each region
      y[MAXD][2] = {0}, //Latitude and Longitude
      yPred[MAXD][2] = {0}, //Prediction of Latitude and Longitude
      numPred[MAXD] = {0},
      
      totmu[MAXL][2] = {0},
      Csigma[MAXL] = {0},
      Asigma = 0,
      Bsigma = 0,
      lambda = 100,  //Penalty Coefficient for \sigma
      pi = 3.1415926;
      
int DocuCoun[MAXD][40] = {0};
    
void Initial_State()
{
    for (int k = 0; k < D; k++){
        //Ramdomly initialize locations
        int loc = floor(dis(gen) * L);
        p[k] = loc;
        for (int u = 0; u < F + 1; u++){
            xuiv[u][loc][x[k][u]]++;
            xuix[u][loc]++;
        }
        
        //Randomly initialize topics
        int len = kxxx[k];
        xxxi[loc] += len;
        for (int l = 0; l < len; l++){
            int topic = floor(dis(gen) * T);
            int v = w[k][l];
            z[k][l] = topic;
            kxjx[k][topic]++;
            xrji[v][topic][loc]++;
            xxji[topic][loc]++;
        }
    }
}

int Sample_Z(int k, int l)
{
    int topic = z[k][l];
    int loc = p[k];
    int v = w[k][l];

    //Remove the current word
    kxjx[k][topic]--;
    xrji[v][topic][loc]--;
    xxji[topic][loc]--;   
    //cout<<kxjx[k][topic]<<' '<<xrji[v][topic][loc]<<' '<<xxji[topic][loc]<<endl;

    //Calculate the probability
    float pr[MAXT] = {0};
    for (int i = 0; i < T; i++){
        //Calculate the sum of the logarithms
        pr[i] = log(kxjx[k][i] + alpha) + log(xrji[v][i][loc] + beta) - log(xxji[i][loc] + V*beta);
    }

    //Calculate the sum
    float maxpr = -100000;
    for (int i = 0; i < T; i++)
        if (pr[i] > maxpr) maxpr = pr[i];
    for (int i = 0; i < T; i++)
        pr[i] -= maxpr;
    pr[0] = exp(pr[0]);
    for (int i = 1; i < T; i++)
        pr[i] = exp(pr[i])+pr[i-1];

    //Sample
    float ran = dis(gen) * pr[T-1];
    for (topic = 0; topic < T; topic++){
        if (ran < pr[topic]+1e-6){
            break;
        }
    }

    //Update the topic assignment
    kxjx[k][topic]++;
    xrji[v][topic][loc]++;
    xxji[topic][loc]++;
    return topic;
}

int Sample_P(int k)
{
    int len = kxxx[k];
    int loc = p[k];
    int tmp[MAXN][3] = {0}; //Count the number of each topic-word pair in the document
    int tmplen = 0, z0, w0, j;
    for (int i = 0; i < len; i++){
        z0 = z[k][i];
        w0 = w[k][i];
        for (j = 0; j < tmplen; j++){
            if (tmp[j][0] == z0 && tmp[j][1] == w0){
                tmp[j][2]++;  //z0-w0 pair has appeared in the document
                break;
            }
        }
        if (j == tmplen){
            tmp[tmplen][0] = z0;
            tmp[tmplen][1] = w0;
            tmp[tmplen][2] = 1;
            tmplen++;
        }
    }

    //Remove the current document
    xxxi[loc] -= len;
    for (int i = 0; i < tmplen; i++){
        int j = tmp[i][0];
        int r = tmp[i][1];
        xrji[r][j][loc] -= tmp[i][2];
    }
    for (int u = 0; u < F + 1; u++){
        xuiv[u][loc][x[k][u]]--;
        xuix[u][loc]--;
    }

    //Calculate the probability
    float pr[MAXL] = {0};
    for (int i = 0; i < L; i++){
        //Calculate the sum of the logarithms
        for (int l = 0; l < len; l++){
            pr[i] += log(xxxi[i]+gamma_+l);
        }
        for (int l = 0; l < tmplen; l++){
            int j = tmp[l][0];
            int r = tmp[l][1];
            float tmp1 = xrji[r][j][i] + beta;
            float tmp2 = xxji[j][i] + V*beta;
            for (int k = 0; k < tmp[l][2]; k++){
                pr[i] += log(tmp1+k);
            }
            for (int k = 0; k < len; k++){
                pr[i] -= log(tmp2+k);
            }
        }
        for (int u = 0; u < F + 1; u++){
            pr[i] += log(xuiv[u][i][x[k][u]] + delta) - log(xuix[u][i] + Cl[u]*delta);
        }
        pr[i] += -log(sigma2[i]) + 
                 -0.5*((y[k][0]-mu[i][0])*(y[k][0]-mu[i][0])+(y[k][1]-mu[i][1])*(y[k][1]-mu[i][1]))/sigma2[i]; 
    }

    //Calculate the sum
    float maxpr = -100000;
    for (int i = 0; i < L; i++)
        if (pr[i] > maxpr) maxpr = pr[i];
    for (int i = 0; i < L; i++)
        pr[i] -= maxpr;
    pr[0] = exp(pr[0]);
    for (int i = 1; i < L; i++)
        pr[i] = exp(pr[i])+pr[i-1];

    //Sample
    float ran = dis(gen) * pr[L-1];
    for (loc = 0; loc < L; loc++){
        if (ran < pr[loc]+1e-6){
            break;
        }
    }

    //Update the location assignment
    xxxi[loc] += len;
    for (int i = 0; i < tmplen; i++){
        int j = tmp[i][0];
        int r = tmp[i][1];
        xrji[r][j][loc] += tmp[i][2];
    }
    for (int u = 0; u < F + 1; u++){
        xuiv[u][loc][x[k][u]]++;
        xuix[u][loc]++;
    }
    
    return loc;
}

//Expectation: Gibbs_Sampling
void Expectation(int step)
{
    for (int i = 0; i < maxIter; i++){
        for (int j = 0; j < D; j++){
            for (int k = 0; k < kxxx[j]; k++){
                z[j][k] = Sample_Z(j, k);
            }
        }
        for (int j = 0; j < D; j++){
            p[j] = Sample_P(j);
            //save some sample results
            if (i >= burnIn){
                int p0 = p[j];
                totmu[p0][0] += y[j][0];
                totmu[p0][1] += y[j][1];
                num[p0]++;
                Csigma[p0] += (y[j][0]-mu[p0][0])*(y[j][0]-mu[p0][0])+(y[j][1]-mu[p0][1])*(y[j][1]-mu[p0][1]);
            }
        }
        if (i%5 == 0){ 
            cout<<i<<' '<<p[0]<<' '<<y[0][0]<<' '<<y[0][1]<<' '<<mu[p[0]][0]<<' '<<mu[p[0]][1]<<endl;
        }
    }
}

//Maximization
void Maximization()
{
    //Update \mu
    for (int i = 0; i < L; i++){
        if (num[i] != 0){
            mu[i][0] = totmu[i][0]/num[i];
            mu[i][1] = totmu[i][1]/num[i];
        }
        else {
            mu[i][0] = 1000;
            mu[i][1] = 1000;
        }
    }
    //Update \sigma
    for (int i = 0; i < L; i++){
        if (num[i] != 0){
            Asigma = lambda*num[i];
            Bsigma = num[i];
            Csigma[i] *= -1.0/3;
            sigma2[i] = (-Bsigma + sqrt(Bsigma*Bsigma-4*Asigma*Csigma[i])) / (2*Asigma);
            Csigma[i] = 0;
        }
        else {
            sigma2[i] = 1;
        }
        if (sigma2[i] < 1e-6){
            mu[i][0] = 1000;
            mu[i][1] = 1000;
            sigma2[i] = 1;
        }
        totmu[i][0] = 0;
        totmu[i][1] = 0;
        num[i] = 0;
    }
}

void Initial_State0()
{
    for (int k = D; k < D0; k++){
        //Ramdomly initialize locations
        int loc = floor(dis(gen) * L);
        p[k] = loc;
        //Randomly initialize topics
        int len = kxxx[k];
        for (int l = 0; l < len; l++){
            int topic = floor(dis(gen) * T);
            z[k][l] = topic;
        }
    }
}

int Sample_Z0(int k, int l)
{
    int loc = p[k];
    int v = w[k][l];

    //Calculate the probability
    float pr[MAXT] = {0};
    for (int i = 0; i < T; i++){
        //Calculate the sum of the logarithms
        pr[i] = log(kxjx[k][i] + alpha) + log(xrji[v][i][loc] + beta) - log(xxji[i][loc] + V*beta);
    }

    //Calculate the sum
    float maxpr = -100000;
    for (int i = 0; i < T; i++)
        if (pr[i] > maxpr) maxpr = pr[i];
    for (int i = 0; i < T; i++)
        pr[i] -= maxpr;
    pr[0] = exp(pr[0]);
    for (int i = 1; i < T; i++)
        pr[i] = exp(pr[i])+pr[i-1];

    //Sample
    int topic;
    float ran = dis(gen) * pr[T-1];
    for (topic = 0; topic < T; topic++){
        if (ran < pr[topic]+1e-6){
            break;
        }
    }

    return topic;
}

int Sample_P0(int k)
{
    int len = kxxx[k];
    // int tmp[MAXN][3] = {0}; //Count the number of each topic-word pair in the document
    // int tmplen = 0, z0, w0, j;
    // for (int i = 0; i < len; i++){
    //     z0 = z[k][i];
    //     w0 = w[k][i];
    //     for (j = 0; j < tmplen; j++){
    //         if (tmp[j][0] == z0 && tmp[j][1] == w0){
    //             tmp[j][2]++;  //z0-w0 pair has appeared in the document
    //             break;
    //         }
    //     }
    //     if (j == tmplen){
    //         tmp[tmplen][0] = z0;
    //         tmp[tmplen][1] = w0;
    //         tmp[tmplen][2] = 1;
    //         tmplen++;
    //     }
    // }

    //Calculate the probability
    float pr[MAXL] = {0};
    for (int i = 0; i < L; i++){
        //Calculate the sum of the logarithms
        pr[i] += log(xxxi[i]+gamma_);
        for (int l = 0; l < len; l++){
            int z0 = z[k][l];
            int w0 = w[k][l];
            pr[i] += log(xrji[w0][z0][i] + beta) - log(xxji[z0][i] + V*beta);
        }
        for (int u = 0; u < F; u++){
            int x0 = x[k][u];
            pr[i] += log(xuiv[u][i][x0] + delta) - log(xuix[u][i] + Cl[u]*delta);
        }
        // No Lat and Lng information in Prediction!!
        // pr[i] += -log(sigma2[i]) + 
        //          -0.5*((y[k][0]-mu[i][0])*(y[k][0]-mu[i][0])+(y[k][1]-mu[i][1])*(y[k][1]-mu[i][1]))/sigma2[i]; 
    }

    //Calculate the sum
    float maxpr = -100000;
    for (int i = 0; i < L; i++)
        if (pr[i] > maxpr) maxpr = pr[i];
    for (int i = 0; i < L; i++)
        pr[i] -= maxpr;
    pr[0] = exp(pr[0]);
    for (int i = 1; i < L; i++)
        pr[i] = exp(pr[i])+pr[i-1];

    //Sample
    int loc;
    float ran = dis(gen) * pr[L-1];
    for (loc = 0; loc < L; loc++){
        if (ran < pr[loc]+1e-6){
            break;
        }
    }
    
    if (loc == L){
        cout<<ran<<' '<<pr[L-1]<<endl;
    }

    return loc;
}

void Expectation0()
{
    for (int i = 0; i < maxIter; i++){
        for (int j = D; j < D0; j++){
            for (int k = 0; k < kxxx[j]; k++){
                z[j][k] = Sample_Z0(j, k);
            }
        }
        for (int j = D; j < D0; j++){
            p[j] = Sample_P0(j);
            //save some sample results
            if (i >= burnIn){
                int p0 = p[j];
                yPred[j][0] += mu[p0][0]/sigma2[p0];
                yPred[j][1] += mu[p0][1]/sigma2[p0];
                numPred[j] += 1/sigma2[p0];
                
                for (int k = 0; k < 35; k++){
                    DocuCoun[j][k] += log(xuiv[F][p0][k] + delta) - log(xuix[F][p0] + 35 * delta);
                }
            }
        }
        if (i%100 == 0) 
            cout<<i<<' '<<p[D]<<endl;
    }
}

void Maximization0()
{
    ofstream fout("result.txt", ios::out);

    //Distance
    float r = 6371;
    float dis = 0, ddis = 0;
    int count = 0;
    for (int i = D; i < D0; i++){
        yPred[i][0] = yPred[i][0]/numPred[i];
        yPred[i][1] = yPred[i][1]/numPred[i];
        
        int maxL = -1000000, maxC;
        for (int j = 0; j < 35; j++){
            if (DocuCoun[i][j] > maxL){
                maxL = DocuCoun[i][j];
                maxC = j;
            }
        }
        if (maxC == x[i][F]) count++;
        
        y[i][0] *= pi/180;
        y[i][1] *= pi/180;
        yPred[i][0] *= pi/180;
        yPred[i][1] *= pi/180;
        ddis = 2*r*asin(sqrt(sin((yPred[i][0]-y[i][0])/2)*sin((yPred[i][0]-y[i][0])/2) 
             + 2*cos(yPred[i][0])*cos(y[i][0])*sin((yPred[i][1]-y[i][1])/2)*sin((yPred[i][1]-y[i][1])/2)));
        dis += ddis;

        fout<<x[i][F]<<' '<<maxC<<' '<<ddis<<endl;
    }

    fout.close();

    cout<<"\nRESULT\n";
    cout<<"Mean Distance Error (km): "<<dis/(D0-D)<<endl;
    cout<<"Accuracy: "<<(float)count/(D0-D)<<endl;
}

void Gibbs_EM()
{
    //Innitialize each variable
    Initial_State();

    //Training - EM
    for (int i = 0; i < EMStep; i++){
        Expectation(i);
        Maximization();
        for (int i = 0; i < L; i++){
            cout<<mu[i][0]<<' '<<mu[i][1]<<' '<<sigma2[i]<<endl;
        }
        cout<<endl;
    }
    
    //Innitialize each variable
    Initial_State0();
    
    //Prediction - EM
    Expectation0();
    Maximization0();
}

int main(int argc, char* argv[])
{    
    ifstream fin1(argv[1], ios::in);  //Training Set
    ifstream fin2(argv[2], ios::in);  //Testing Set

    fin1 >> D;
    fin2 >> D0;
    D0 = D + D0;

    if (argc >= 4){
        L = atoi(argv[3]);
    }

    string st = "./Dataset/Kmeans" + to_string(L) + ".txt";
    ifstream fin3(st, ios::in);  //Kmeans File

    if (argc >= 5){
        T = atoi(argv[4]);
    }

    int pad;
    alpha = 50.0/(L*T);
    for (int i = 0; i < D; i++){
        for (int j = 0; j < F + 1; j++) 
            fin1>>x[i][j];            //Features & Country
        fin1>>y[i][0]>>y[i][1];       //Latitude and Longitude
        fin1>>kxxx[i];
        for (int j = 0; j < kxxx[i]; j++)
            fin1>>w[i][j];            //Text
    }
    fin1.close();

    for (int i = D+1; i < D0; i++){
        for (int j = 0; j < F + 1; j++) 
            fin2>>x[i][j];            //Features & Country 
        fin2>>y[i][0]>>y[i][1];       //Latitude and Longitude
        fin2>>kxxx[i];
        for (int j = 0; j < kxxx[i]; j++)
            fin2>>w[i][j];            //Text
    }
    fin2.close();

    for (int i = 0; i < L; i++){
        fin3>>mu[i][0]>>mu[i][1];    //Center of region
        sigma2[i] = 1;               //Variance of region
    }
    fin3.close();
    
    Gibbs_EM();

    return 0;
}
