#pragma version(1)
#pragma rs java_package_name(com.example.mmota.squeezenet_dse)
//#pragma rs_fp_relaxed
//#pragma java_package_name(thesis.myapplication)
#pragma rs_fp_imprecise


int32_t K;
int32_t Wout;
int32_t Hout;
int32_t Win;
int32_t Hin;
int32_t M;
int32_t N;
int32_t S;
int32_t pad;
int32_t N_new;
int32_t mtmd;
int32_t offset;
int32_t parallelOFMs;

rs_allocation weight;
rs_allocation in;

rs_allocation bias;

rs_allocation output;
float __attribute__ ((kernel)) avgPool(int32_t x) {
    int32_t w = (x) % Wout;
    int32_t h = ((x) / Wout) % Hout;
    int32_t n = (x / (4 * Wout * Hout)) * 4;
    int32_t m = x / (Wout * Hout);

    int32_t ww, hh;
    float out;
    int32_t Wstart = w * S;
    int32_t Hstart = h * S;
    int32_t Wend = Wstart + K;
    int32_t Hend = Hstart + K;
    float4 sum;
    sum.x = 0;
    sum.y = 0;
    sum.z = 0;
    sum.w = 0;

    for (ww = Wstart; ww < Wend; ww++){
        for (hh = Hstart; hh < Hend; hh++){
            sum = sum + rsGetElementAt_float4(in, ww + Win * hh + Win * Hin * m);
        }
    }
    sum = sum / (K * K);

    int32_t idx = (x % (Wout * Hout));
    idx = 4 * idx;
    rsSetElementAt_float(output, sum.x, (idx + Wout * Hout * 4 * m));
    rsSetElementAt_float(output, sum.y, (idx + 1 + Wout * Hout * 4 * m));
    rsSetElementAt_float(output, sum.z, (idx + 2 + Wout * Hout * 4 * m));
    rsSetElementAt_float(output, sum.w, (idx + 3 + Wout * Hout * 4 * m));

    //return out;
}
//float __attribute__ ((kernel)) lrn (int32_t x) {
//    int32_t w = (x) % Wout;
//    int32_t h = ((x) / Wout) % Hout;
//    int32_t n = (x / (4 * Wout * Hout)) * 4; // BOX ID
//    int32_t m = x / (Wout * Hout); //Layer ID

//    double tmp = 0;
//    float lrnOut = 0;
//    int Mstart, Mend, int z;
//    tmp = pow(rsGetElementAt_float(in, w + Win * h + Win * Hin * n), 2);
//    Mstart = 0 > (n - localSize / 2) ? 0 : (n - localSize / 2);
//    Mend = (n + localSize / 2 + 1) < N ? (n + localSize / 2 + 1) : N;
//    for (z = Mstart; z < Mend; z++){
//        out[w + Win * h + Win * Hin * z] += tmp;
//    }
//}
float __attribute__ ((kernel)) maxPoolNew (int32_t x) {
    int32_t w = (x) % Wout;
    int32_t h = ((x) / Wout) % Hout;
    int32_t n = (x / (4 * Wout * Hout)) * 4;
    int32_t m = x / (Wout * Hout);

    w = w - 1;
    h = h - 1;

    int32_t ww, hh;
    float out;
    int32_t Wstart = w * S;
    int32_t Hstart = h * S;
    int32_t Wend = Wstart + K < Win ? Wstart + K : Win;
    int32_t Hend = Hstart + K < Hin ? Hstart + K : Hin;
    //ToDo: Replace this with MAX_FLT
    float4 maxValue;

    maxValue.x = -3.402823e10;
    maxValue.y = -3.402823e10;
    maxValue.z = -3.402823e10;
    maxValue.w = -3.402823e10;

    for (ww = Wstart; ww < Wend; ww++){
        for (hh = Hstart; hh < Hend; hh++){
            if (ww < 0 || ww >= Win || hh <0 || hh >= Hin) continue;
            maxValue = fmax(maxValue, rsGetElementAt_float4(in, ww + Win * hh + Win * Hin * m));
        }
    }

    int32_t idx = (x % (Wout * Hout));
    idx = 4 * idx;
    rsSetElementAt_float(output, maxValue.x, (idx + Wout * Hout * 4 * m));
    rsSetElementAt_float(output, maxValue.y, (idx + 1 + Wout * Hout * 4 * m));
    rsSetElementAt_float(output, maxValue.z, (idx + 2 + Wout * Hout * 4 * m));
    rsSetElementAt_float(output, maxValue.w, (idx + 3 + Wout * Hout * 4 * m));

    out = maxValue.x;

    return out;
}
float __attribute__ ((kernel)) maxPool (int32_t x) {
    int32_t w = (x) % Wout;
    int32_t h = ((x) / Wout) % Hout;
    int32_t n = (x / (4 * Wout * Hout)) * 4;
    int32_t m = x / (Wout * Hout);


    int32_t ww, hh;
    float out;
    int32_t Wstart = w * S;
    int32_t Hstart = h * S;
    int32_t Wend = Wstart + K < Win ? Wstart + K : Win;
    int32_t Hend = Hstart + K < Hin ? Hstart + K : Hin;
    //ToDo: Replace this with MAX_FLT
    float4 maxValue;

    maxValue.x = -3.402823e10;
    maxValue.y = -3.402823e10;
    maxValue.z = -3.402823e10;
    maxValue.w = -3.402823e10;

    for (ww = Wstart; ww < Wend; ww++){
        for (hh = Hstart; hh < Hend; hh++){
            maxValue = fmax(maxValue, rsGetElementAt_float4(in, ww + Win * hh + Win * Hin * m));
        }
    }

    int32_t idx = (x % (Wout * Hout));
    idx = 4 * idx;
    rsSetElementAt_float(output, maxValue.x, (idx + Wout * Hout * 4 * m));
    rsSetElementAt_float(output, maxValue.y, (idx + 1 + Wout * Hout * 4 * m));
    rsSetElementAt_float(output, maxValue.z, (idx + 2 + Wout * Hout * 4 * m));
    rsSetElementAt_float(output, maxValue.w, (idx + 3 + Wout * Hout * 4 * m));

    out = maxValue.x;

    return out;
}

float __attribute__((kernel)) reshape(uint32_t x) {
    int32_t w = x % Wout;
    int32_t h = ((x - w) / Wout) % Hout;

    float4 out;
    out.x = rsGetElementAt_float(in, w + Wout * h);
    out.y = rsGetElementAt_float(in, w + Wout * h + (Wout * Hout * 1));
    out.z = rsGetElementAt_float(in, w + Wout * h + (Wout * Hout * 2));
    out.w = 0;

    rsSetElementAt_float4(output, out, x);
}
float __attribute__((kernel)) normalToVectorized(uint32_t x) {
// Converts normal array to a vectorized format.
    int32_t w = x % Wout;
    int32_t h = ((x - w) / Wout) % Hout;

    float4 out;
    int i = 0;
    for (i = 0; i < (N / 4); i++) {
        int base = 4 * i * Wout * Hout;
        out.x = rsGetElementAt_float(in, w + Wout * h + base);
        out.y = rsGetElementAt_float(in, w + Wout * h + base + (Wout * Hout * 1));
        out.z = rsGetElementAt_float(in, w + Wout * h + base + (Wout * Hout * 2));
        out.w = rsGetElementAt_float(in, w + Wout * h + base + (Wout * Hout * 3));
        rsSetElementAt_float4(output, out, i * Wout * Hout + x);
    }

}

float __attribute__((kernel)) conv_1(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

}
float __attribute__((kernel)) conv_2(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

}


float __attribute__((kernel)) conv_4(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

}
float __attribute__((kernel)) conv_5(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

}

float __attribute__((kernel)) conv_6(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

}
float __attribute__((kernel)) conv_7(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

}

float __attribute__((kernel)) conv_8(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

}
float __attribute__((kernel)) conv_9(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

}

float __attribute__((kernel)) conv_10(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

}

float __attribute__((kernel)) conv_12(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

}
float __attribute__((kernel)) conv_13(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

}
float __attribute__((kernel)) conv_14(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

}

float __attribute__((kernel)) conv_16(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

}
float __attribute__((kernel)) conv_18(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

}

float __attribute__((kernel)) conv_20(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

}
float __attribute__((kernel)) conv_24(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

}
float __attribute__((kernel)) conv_26(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

}

float __attribute__((kernel)) conv_28(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);
	float out26 = rsGetElementAt_float(bias, m + 26 * parallelOFMs);
	float out27 = rsGetElementAt_float(bias, m + 27 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

				float4 wght26 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 26 * parallelOFMs)));
				out26 += dot(ifm, wght26);

				float4 wght27 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 27 * parallelOFMs)));
				out27 += dot(ifm, wght27);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

	out26 = fmax(0.0f, out26);
	rsSetElementAt_float(output, out26, base3 + 26 * base4);

	out27 = fmax(0.0f, out27);
	rsSetElementAt_float(output, out27, base3 + 27 * base4);

}
float __attribute__((kernel)) conv_32(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);
	float out26 = rsGetElementAt_float(bias, m + 26 * parallelOFMs);
	float out27 = rsGetElementAt_float(bias, m + 27 * parallelOFMs);
	float out28 = rsGetElementAt_float(bias, m + 28 * parallelOFMs);
	float out29 = rsGetElementAt_float(bias, m + 29 * parallelOFMs);
	float out30 = rsGetElementAt_float(bias, m + 30 * parallelOFMs);
	float out31 = rsGetElementAt_float(bias, m + 31 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

				float4 wght26 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 26 * parallelOFMs)));
				out26 += dot(ifm, wght26);

				float4 wght27 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 27 * parallelOFMs)));
				out27 += dot(ifm, wght27);

				float4 wght28 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 28 * parallelOFMs)));
				out28 += dot(ifm, wght28);

				float4 wght29 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 29 * parallelOFMs)));
				out29 += dot(ifm, wght29);

				float4 wght30 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 30 * parallelOFMs)));
				out30 += dot(ifm, wght30);

				float4 wght31 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 31 * parallelOFMs)));
				out31 += dot(ifm, wght31);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

	out26 = fmax(0.0f, out26);
	rsSetElementAt_float(output, out26, base3 + 26 * base4);

	out27 = fmax(0.0f, out27);
	rsSetElementAt_float(output, out27, base3 + 27 * base4);

	out28 = fmax(0.0f, out28);
	rsSetElementAt_float(output, out28, base3 + 28 * base4);

	out29 = fmax(0.0f, out29);
	rsSetElementAt_float(output, out29, base3 + 29 * base4);

	out30 = fmax(0.0f, out30);
	rsSetElementAt_float(output, out30, base3 + 30 * base4);

	out31 = fmax(0.0f, out31);
	rsSetElementAt_float(output, out31, base3 + 31 * base4);

}
float __attribute__((kernel)) conv_36(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);
	float out26 = rsGetElementAt_float(bias, m + 26 * parallelOFMs);
	float out27 = rsGetElementAt_float(bias, m + 27 * parallelOFMs);
	float out28 = rsGetElementAt_float(bias, m + 28 * parallelOFMs);
	float out29 = rsGetElementAt_float(bias, m + 29 * parallelOFMs);
	float out30 = rsGetElementAt_float(bias, m + 30 * parallelOFMs);
	float out31 = rsGetElementAt_float(bias, m + 31 * parallelOFMs);
	float out32 = rsGetElementAt_float(bias, m + 32 * parallelOFMs);
	float out33 = rsGetElementAt_float(bias, m + 33 * parallelOFMs);
	float out34 = rsGetElementAt_float(bias, m + 34 * parallelOFMs);
	float out35 = rsGetElementAt_float(bias, m + 35 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

				float4 wght26 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 26 * parallelOFMs)));
				out26 += dot(ifm, wght26);

				float4 wght27 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 27 * parallelOFMs)));
				out27 += dot(ifm, wght27);

				float4 wght28 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 28 * parallelOFMs)));
				out28 += dot(ifm, wght28);

				float4 wght29 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 29 * parallelOFMs)));
				out29 += dot(ifm, wght29);

				float4 wght30 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 30 * parallelOFMs)));
				out30 += dot(ifm, wght30);

				float4 wght31 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 31 * parallelOFMs)));
				out31 += dot(ifm, wght31);

				float4 wght32 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 32 * parallelOFMs)));
				out32 += dot(ifm, wght32);

				float4 wght33 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 33 * parallelOFMs)));
				out33 += dot(ifm, wght33);

				float4 wght34 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 34 * parallelOFMs)));
				out34 += dot(ifm, wght34);

				float4 wght35 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 35 * parallelOFMs)));
				out35 += dot(ifm, wght35);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

	out26 = fmax(0.0f, out26);
	rsSetElementAt_float(output, out26, base3 + 26 * base4);

	out27 = fmax(0.0f, out27);
	rsSetElementAt_float(output, out27, base3 + 27 * base4);

	out28 = fmax(0.0f, out28);
	rsSetElementAt_float(output, out28, base3 + 28 * base4);

	out29 = fmax(0.0f, out29);
	rsSetElementAt_float(output, out29, base3 + 29 * base4);

	out30 = fmax(0.0f, out30);
	rsSetElementAt_float(output, out30, base3 + 30 * base4);

	out31 = fmax(0.0f, out31);
	rsSetElementAt_float(output, out31, base3 + 31 * base4);

	out32 = fmax(0.0f, out32);
	rsSetElementAt_float(output, out32, base3 + 32 * base4);

	out33 = fmax(0.0f, out33);
	rsSetElementAt_float(output, out33, base3 + 33 * base4);

	out34 = fmax(0.0f, out34);
	rsSetElementAt_float(output, out34, base3 + 34 * base4);

	out35 = fmax(0.0f, out35);
	rsSetElementAt_float(output, out35, base3 + 35 * base4);

}
float __attribute__((kernel)) conv_40(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);
	float out26 = rsGetElementAt_float(bias, m + 26 * parallelOFMs);
	float out27 = rsGetElementAt_float(bias, m + 27 * parallelOFMs);
	float out28 = rsGetElementAt_float(bias, m + 28 * parallelOFMs);
	float out29 = rsGetElementAt_float(bias, m + 29 * parallelOFMs);
	float out30 = rsGetElementAt_float(bias, m + 30 * parallelOFMs);
	float out31 = rsGetElementAt_float(bias, m + 31 * parallelOFMs);
	float out32 = rsGetElementAt_float(bias, m + 32 * parallelOFMs);
	float out33 = rsGetElementAt_float(bias, m + 33 * parallelOFMs);
	float out34 = rsGetElementAt_float(bias, m + 34 * parallelOFMs);
	float out35 = rsGetElementAt_float(bias, m + 35 * parallelOFMs);
	float out36 = rsGetElementAt_float(bias, m + 36 * parallelOFMs);
	float out37 = rsGetElementAt_float(bias, m + 37 * parallelOFMs);
	float out38 = rsGetElementAt_float(bias, m + 38 * parallelOFMs);
	float out39 = rsGetElementAt_float(bias, m + 39 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

				float4 wght26 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 26 * parallelOFMs)));
				out26 += dot(ifm, wght26);

				float4 wght27 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 27 * parallelOFMs)));
				out27 += dot(ifm, wght27);

				float4 wght28 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 28 * parallelOFMs)));
				out28 += dot(ifm, wght28);

				float4 wght29 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 29 * parallelOFMs)));
				out29 += dot(ifm, wght29);

				float4 wght30 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 30 * parallelOFMs)));
				out30 += dot(ifm, wght30);

				float4 wght31 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 31 * parallelOFMs)));
				out31 += dot(ifm, wght31);

				float4 wght32 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 32 * parallelOFMs)));
				out32 += dot(ifm, wght32);

				float4 wght33 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 33 * parallelOFMs)));
				out33 += dot(ifm, wght33);

				float4 wght34 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 34 * parallelOFMs)));
				out34 += dot(ifm, wght34);

				float4 wght35 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 35 * parallelOFMs)));
				out35 += dot(ifm, wght35);

				float4 wght36 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 36 * parallelOFMs)));
				out36 += dot(ifm, wght36);

				float4 wght37 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 37 * parallelOFMs)));
				out37 += dot(ifm, wght37);

				float4 wght38 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 38 * parallelOFMs)));
				out38 += dot(ifm, wght38);

				float4 wght39 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 39 * parallelOFMs)));
				out39 += dot(ifm, wght39);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

	out26 = fmax(0.0f, out26);
	rsSetElementAt_float(output, out26, base3 + 26 * base4);

	out27 = fmax(0.0f, out27);
	rsSetElementAt_float(output, out27, base3 + 27 * base4);

	out28 = fmax(0.0f, out28);
	rsSetElementAt_float(output, out28, base3 + 28 * base4);

	out29 = fmax(0.0f, out29);
	rsSetElementAt_float(output, out29, base3 + 29 * base4);

	out30 = fmax(0.0f, out30);
	rsSetElementAt_float(output, out30, base3 + 30 * base4);

	out31 = fmax(0.0f, out31);
	rsSetElementAt_float(output, out31, base3 + 31 * base4);

	out32 = fmax(0.0f, out32);
	rsSetElementAt_float(output, out32, base3 + 32 * base4);

	out33 = fmax(0.0f, out33);
	rsSetElementAt_float(output, out33, base3 + 33 * base4);

	out34 = fmax(0.0f, out34);
	rsSetElementAt_float(output, out34, base3 + 34 * base4);

	out35 = fmax(0.0f, out35);
	rsSetElementAt_float(output, out35, base3 + 35 * base4);

	out36 = fmax(0.0f, out36);
	rsSetElementAt_float(output, out36, base3 + 36 * base4);

	out37 = fmax(0.0f, out37);
	rsSetElementAt_float(output, out37, base3 + 37 * base4);

	out38 = fmax(0.0f, out38);
	rsSetElementAt_float(output, out38, base3 + 38 * base4);

	out39 = fmax(0.0f, out39);
	rsSetElementAt_float(output, out39, base3 + 39 * base4);

}
float __attribute__((kernel)) conv_44(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);
	float out26 = rsGetElementAt_float(bias, m + 26 * parallelOFMs);
	float out27 = rsGetElementAt_float(bias, m + 27 * parallelOFMs);
	float out28 = rsGetElementAt_float(bias, m + 28 * parallelOFMs);
	float out29 = rsGetElementAt_float(bias, m + 29 * parallelOFMs);
	float out30 = rsGetElementAt_float(bias, m + 30 * parallelOFMs);
	float out31 = rsGetElementAt_float(bias, m + 31 * parallelOFMs);
	float out32 = rsGetElementAt_float(bias, m + 32 * parallelOFMs);
	float out33 = rsGetElementAt_float(bias, m + 33 * parallelOFMs);
	float out34 = rsGetElementAt_float(bias, m + 34 * parallelOFMs);
	float out35 = rsGetElementAt_float(bias, m + 35 * parallelOFMs);
	float out36 = rsGetElementAt_float(bias, m + 36 * parallelOFMs);
	float out37 = rsGetElementAt_float(bias, m + 37 * parallelOFMs);
	float out38 = rsGetElementAt_float(bias, m + 38 * parallelOFMs);
	float out39 = rsGetElementAt_float(bias, m + 39 * parallelOFMs);
	float out40 = rsGetElementAt_float(bias, m + 40 * parallelOFMs);
	float out41 = rsGetElementAt_float(bias, m + 41 * parallelOFMs);
	float out42 = rsGetElementAt_float(bias, m + 42 * parallelOFMs);
	float out43 = rsGetElementAt_float(bias, m + 43 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

				float4 wght26 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 26 * parallelOFMs)));
				out26 += dot(ifm, wght26);

				float4 wght27 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 27 * parallelOFMs)));
				out27 += dot(ifm, wght27);

				float4 wght28 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 28 * parallelOFMs)));
				out28 += dot(ifm, wght28);

				float4 wght29 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 29 * parallelOFMs)));
				out29 += dot(ifm, wght29);

				float4 wght30 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 30 * parallelOFMs)));
				out30 += dot(ifm, wght30);

				float4 wght31 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 31 * parallelOFMs)));
				out31 += dot(ifm, wght31);

				float4 wght32 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 32 * parallelOFMs)));
				out32 += dot(ifm, wght32);

				float4 wght33 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 33 * parallelOFMs)));
				out33 += dot(ifm, wght33);

				float4 wght34 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 34 * parallelOFMs)));
				out34 += dot(ifm, wght34);

				float4 wght35 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 35 * parallelOFMs)));
				out35 += dot(ifm, wght35);

				float4 wght36 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 36 * parallelOFMs)));
				out36 += dot(ifm, wght36);

				float4 wght37 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 37 * parallelOFMs)));
				out37 += dot(ifm, wght37);

				float4 wght38 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 38 * parallelOFMs)));
				out38 += dot(ifm, wght38);

				float4 wght39 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 39 * parallelOFMs)));
				out39 += dot(ifm, wght39);

				float4 wght40 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 40 * parallelOFMs)));
				out40 += dot(ifm, wght40);

				float4 wght41 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 41 * parallelOFMs)));
				out41 += dot(ifm, wght41);

				float4 wght42 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 42 * parallelOFMs)));
				out42 += dot(ifm, wght42);

				float4 wght43 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 43 * parallelOFMs)));
				out43 += dot(ifm, wght43);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

	out26 = fmax(0.0f, out26);
	rsSetElementAt_float(output, out26, base3 + 26 * base4);

	out27 = fmax(0.0f, out27);
	rsSetElementAt_float(output, out27, base3 + 27 * base4);

	out28 = fmax(0.0f, out28);
	rsSetElementAt_float(output, out28, base3 + 28 * base4);

	out29 = fmax(0.0f, out29);
	rsSetElementAt_float(output, out29, base3 + 29 * base4);

	out30 = fmax(0.0f, out30);
	rsSetElementAt_float(output, out30, base3 + 30 * base4);

	out31 = fmax(0.0f, out31);
	rsSetElementAt_float(output, out31, base3 + 31 * base4);

	out32 = fmax(0.0f, out32);
	rsSetElementAt_float(output, out32, base3 + 32 * base4);

	out33 = fmax(0.0f, out33);
	rsSetElementAt_float(output, out33, base3 + 33 * base4);

	out34 = fmax(0.0f, out34);
	rsSetElementAt_float(output, out34, base3 + 34 * base4);

	out35 = fmax(0.0f, out35);
	rsSetElementAt_float(output, out35, base3 + 35 * base4);

	out36 = fmax(0.0f, out36);
	rsSetElementAt_float(output, out36, base3 + 36 * base4);

	out37 = fmax(0.0f, out37);
	rsSetElementAt_float(output, out37, base3 + 37 * base4);

	out38 = fmax(0.0f, out38);
	rsSetElementAt_float(output, out38, base3 + 38 * base4);

	out39 = fmax(0.0f, out39);
	rsSetElementAt_float(output, out39, base3 + 39 * base4);

	out40 = fmax(0.0f, out40);
	rsSetElementAt_float(output, out40, base3 + 40 * base4);

	out41 = fmax(0.0f, out41);
	rsSetElementAt_float(output, out41, base3 + 41 * base4);

	out42 = fmax(0.0f, out42);
	rsSetElementAt_float(output, out42, base3 + 42 * base4);

	out43 = fmax(0.0f, out43);
	rsSetElementAt_float(output, out43, base3 + 43 * base4);

}
float __attribute__((kernel)) conv_48(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);
	float out26 = rsGetElementAt_float(bias, m + 26 * parallelOFMs);
	float out27 = rsGetElementAt_float(bias, m + 27 * parallelOFMs);
	float out28 = rsGetElementAt_float(bias, m + 28 * parallelOFMs);
	float out29 = rsGetElementAt_float(bias, m + 29 * parallelOFMs);
	float out30 = rsGetElementAt_float(bias, m + 30 * parallelOFMs);
	float out31 = rsGetElementAt_float(bias, m + 31 * parallelOFMs);
	float out32 = rsGetElementAt_float(bias, m + 32 * parallelOFMs);
	float out33 = rsGetElementAt_float(bias, m + 33 * parallelOFMs);
	float out34 = rsGetElementAt_float(bias, m + 34 * parallelOFMs);
	float out35 = rsGetElementAt_float(bias, m + 35 * parallelOFMs);
	float out36 = rsGetElementAt_float(bias, m + 36 * parallelOFMs);
	float out37 = rsGetElementAt_float(bias, m + 37 * parallelOFMs);
	float out38 = rsGetElementAt_float(bias, m + 38 * parallelOFMs);
	float out39 = rsGetElementAt_float(bias, m + 39 * parallelOFMs);
	float out40 = rsGetElementAt_float(bias, m + 40 * parallelOFMs);
	float out41 = rsGetElementAt_float(bias, m + 41 * parallelOFMs);
	float out42 = rsGetElementAt_float(bias, m + 42 * parallelOFMs);
	float out43 = rsGetElementAt_float(bias, m + 43 * parallelOFMs);
	float out44 = rsGetElementAt_float(bias, m + 44 * parallelOFMs);
	float out45 = rsGetElementAt_float(bias, m + 45 * parallelOFMs);
	float out46 = rsGetElementAt_float(bias, m + 46 * parallelOFMs);
	float out47 = rsGetElementAt_float(bias, m + 47 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

				float4 wght26 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 26 * parallelOFMs)));
				out26 += dot(ifm, wght26);

				float4 wght27 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 27 * parallelOFMs)));
				out27 += dot(ifm, wght27);

				float4 wght28 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 28 * parallelOFMs)));
				out28 += dot(ifm, wght28);

				float4 wght29 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 29 * parallelOFMs)));
				out29 += dot(ifm, wght29);

				float4 wght30 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 30 * parallelOFMs)));
				out30 += dot(ifm, wght30);

				float4 wght31 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 31 * parallelOFMs)));
				out31 += dot(ifm, wght31);

				float4 wght32 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 32 * parallelOFMs)));
				out32 += dot(ifm, wght32);

				float4 wght33 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 33 * parallelOFMs)));
				out33 += dot(ifm, wght33);

				float4 wght34 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 34 * parallelOFMs)));
				out34 += dot(ifm, wght34);

				float4 wght35 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 35 * parallelOFMs)));
				out35 += dot(ifm, wght35);

				float4 wght36 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 36 * parallelOFMs)));
				out36 += dot(ifm, wght36);

				float4 wght37 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 37 * parallelOFMs)));
				out37 += dot(ifm, wght37);

				float4 wght38 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 38 * parallelOFMs)));
				out38 += dot(ifm, wght38);

				float4 wght39 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 39 * parallelOFMs)));
				out39 += dot(ifm, wght39);

				float4 wght40 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 40 * parallelOFMs)));
				out40 += dot(ifm, wght40);

				float4 wght41 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 41 * parallelOFMs)));
				out41 += dot(ifm, wght41);

				float4 wght42 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 42 * parallelOFMs)));
				out42 += dot(ifm, wght42);

				float4 wght43 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 43 * parallelOFMs)));
				out43 += dot(ifm, wght43);

				float4 wght44 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 44 * parallelOFMs)));
				out44 += dot(ifm, wght44);

				float4 wght45 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 45 * parallelOFMs)));
				out45 += dot(ifm, wght45);

				float4 wght46 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 46 * parallelOFMs)));
				out46 += dot(ifm, wght46);

				float4 wght47 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 47 * parallelOFMs)));
				out47 += dot(ifm, wght47);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

	out26 = fmax(0.0f, out26);
	rsSetElementAt_float(output, out26, base3 + 26 * base4);

	out27 = fmax(0.0f, out27);
	rsSetElementAt_float(output, out27, base3 + 27 * base4);

	out28 = fmax(0.0f, out28);
	rsSetElementAt_float(output, out28, base3 + 28 * base4);

	out29 = fmax(0.0f, out29);
	rsSetElementAt_float(output, out29, base3 + 29 * base4);

	out30 = fmax(0.0f, out30);
	rsSetElementAt_float(output, out30, base3 + 30 * base4);

	out31 = fmax(0.0f, out31);
	rsSetElementAt_float(output, out31, base3 + 31 * base4);

	out32 = fmax(0.0f, out32);
	rsSetElementAt_float(output, out32, base3 + 32 * base4);

	out33 = fmax(0.0f, out33);
	rsSetElementAt_float(output, out33, base3 + 33 * base4);

	out34 = fmax(0.0f, out34);
	rsSetElementAt_float(output, out34, base3 + 34 * base4);

	out35 = fmax(0.0f, out35);
	rsSetElementAt_float(output, out35, base3 + 35 * base4);

	out36 = fmax(0.0f, out36);
	rsSetElementAt_float(output, out36, base3 + 36 * base4);

	out37 = fmax(0.0f, out37);
	rsSetElementAt_float(output, out37, base3 + 37 * base4);

	out38 = fmax(0.0f, out38);
	rsSetElementAt_float(output, out38, base3 + 38 * base4);

	out39 = fmax(0.0f, out39);
	rsSetElementAt_float(output, out39, base3 + 39 * base4);

	out40 = fmax(0.0f, out40);
	rsSetElementAt_float(output, out40, base3 + 40 * base4);

	out41 = fmax(0.0f, out41);
	rsSetElementAt_float(output, out41, base3 + 41 * base4);

	out42 = fmax(0.0f, out42);
	rsSetElementAt_float(output, out42, base3 + 42 * base4);

	out43 = fmax(0.0f, out43);
	rsSetElementAt_float(output, out43, base3 + 43 * base4);

	out44 = fmax(0.0f, out44);
	rsSetElementAt_float(output, out44, base3 + 44 * base4);

	out45 = fmax(0.0f, out45);
	rsSetElementAt_float(output, out45, base3 + 45 * base4);

	out46 = fmax(0.0f, out46);
	rsSetElementAt_float(output, out46, base3 + 46 * base4);

	out47 = fmax(0.0f, out47);
	rsSetElementAt_float(output, out47, base3 + 47 * base4);

}
float __attribute__((kernel)) conv_52(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);
	float out26 = rsGetElementAt_float(bias, m + 26 * parallelOFMs);
	float out27 = rsGetElementAt_float(bias, m + 27 * parallelOFMs);
	float out28 = rsGetElementAt_float(bias, m + 28 * parallelOFMs);
	float out29 = rsGetElementAt_float(bias, m + 29 * parallelOFMs);
	float out30 = rsGetElementAt_float(bias, m + 30 * parallelOFMs);
	float out31 = rsGetElementAt_float(bias, m + 31 * parallelOFMs);
	float out32 = rsGetElementAt_float(bias, m + 32 * parallelOFMs);
	float out33 = rsGetElementAt_float(bias, m + 33 * parallelOFMs);
	float out34 = rsGetElementAt_float(bias, m + 34 * parallelOFMs);
	float out35 = rsGetElementAt_float(bias, m + 35 * parallelOFMs);
	float out36 = rsGetElementAt_float(bias, m + 36 * parallelOFMs);
	float out37 = rsGetElementAt_float(bias, m + 37 * parallelOFMs);
	float out38 = rsGetElementAt_float(bias, m + 38 * parallelOFMs);
	float out39 = rsGetElementAt_float(bias, m + 39 * parallelOFMs);
	float out40 = rsGetElementAt_float(bias, m + 40 * parallelOFMs);
	float out41 = rsGetElementAt_float(bias, m + 41 * parallelOFMs);
	float out42 = rsGetElementAt_float(bias, m + 42 * parallelOFMs);
	float out43 = rsGetElementAt_float(bias, m + 43 * parallelOFMs);
	float out44 = rsGetElementAt_float(bias, m + 44 * parallelOFMs);
	float out45 = rsGetElementAt_float(bias, m + 45 * parallelOFMs);
	float out46 = rsGetElementAt_float(bias, m + 46 * parallelOFMs);
	float out47 = rsGetElementAt_float(bias, m + 47 * parallelOFMs);
	float out48 = rsGetElementAt_float(bias, m + 48 * parallelOFMs);
	float out49 = rsGetElementAt_float(bias, m + 49 * parallelOFMs);
	float out50 = rsGetElementAt_float(bias, m + 50 * parallelOFMs);
	float out51 = rsGetElementAt_float(bias, m + 51 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

				float4 wght26 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 26 * parallelOFMs)));
				out26 += dot(ifm, wght26);

				float4 wght27 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 27 * parallelOFMs)));
				out27 += dot(ifm, wght27);

				float4 wght28 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 28 * parallelOFMs)));
				out28 += dot(ifm, wght28);

				float4 wght29 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 29 * parallelOFMs)));
				out29 += dot(ifm, wght29);

				float4 wght30 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 30 * parallelOFMs)));
				out30 += dot(ifm, wght30);

				float4 wght31 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 31 * parallelOFMs)));
				out31 += dot(ifm, wght31);

				float4 wght32 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 32 * parallelOFMs)));
				out32 += dot(ifm, wght32);

				float4 wght33 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 33 * parallelOFMs)));
				out33 += dot(ifm, wght33);

				float4 wght34 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 34 * parallelOFMs)));
				out34 += dot(ifm, wght34);

				float4 wght35 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 35 * parallelOFMs)));
				out35 += dot(ifm, wght35);

				float4 wght36 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 36 * parallelOFMs)));
				out36 += dot(ifm, wght36);

				float4 wght37 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 37 * parallelOFMs)));
				out37 += dot(ifm, wght37);

				float4 wght38 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 38 * parallelOFMs)));
				out38 += dot(ifm, wght38);

				float4 wght39 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 39 * parallelOFMs)));
				out39 += dot(ifm, wght39);

				float4 wght40 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 40 * parallelOFMs)));
				out40 += dot(ifm, wght40);

				float4 wght41 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 41 * parallelOFMs)));
				out41 += dot(ifm, wght41);

				float4 wght42 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 42 * parallelOFMs)));
				out42 += dot(ifm, wght42);

				float4 wght43 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 43 * parallelOFMs)));
				out43 += dot(ifm, wght43);

				float4 wght44 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 44 * parallelOFMs)));
				out44 += dot(ifm, wght44);

				float4 wght45 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 45 * parallelOFMs)));
				out45 += dot(ifm, wght45);

				float4 wght46 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 46 * parallelOFMs)));
				out46 += dot(ifm, wght46);

				float4 wght47 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 47 * parallelOFMs)));
				out47 += dot(ifm, wght47);

				float4 wght48 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 48 * parallelOFMs)));
				out48 += dot(ifm, wght48);

				float4 wght49 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 49 * parallelOFMs)));
				out49 += dot(ifm, wght49);

				float4 wght50 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 50 * parallelOFMs)));
				out50 += dot(ifm, wght50);

				float4 wght51 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 51 * parallelOFMs)));
				out51 += dot(ifm, wght51);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

	out26 = fmax(0.0f, out26);
	rsSetElementAt_float(output, out26, base3 + 26 * base4);

	out27 = fmax(0.0f, out27);
	rsSetElementAt_float(output, out27, base3 + 27 * base4);

	out28 = fmax(0.0f, out28);
	rsSetElementAt_float(output, out28, base3 + 28 * base4);

	out29 = fmax(0.0f, out29);
	rsSetElementAt_float(output, out29, base3 + 29 * base4);

	out30 = fmax(0.0f, out30);
	rsSetElementAt_float(output, out30, base3 + 30 * base4);

	out31 = fmax(0.0f, out31);
	rsSetElementAt_float(output, out31, base3 + 31 * base4);

	out32 = fmax(0.0f, out32);
	rsSetElementAt_float(output, out32, base3 + 32 * base4);

	out33 = fmax(0.0f, out33);
	rsSetElementAt_float(output, out33, base3 + 33 * base4);

	out34 = fmax(0.0f, out34);
	rsSetElementAt_float(output, out34, base3 + 34 * base4);

	out35 = fmax(0.0f, out35);
	rsSetElementAt_float(output, out35, base3 + 35 * base4);

	out36 = fmax(0.0f, out36);
	rsSetElementAt_float(output, out36, base3 + 36 * base4);

	out37 = fmax(0.0f, out37);
	rsSetElementAt_float(output, out37, base3 + 37 * base4);

	out38 = fmax(0.0f, out38);
	rsSetElementAt_float(output, out38, base3 + 38 * base4);

	out39 = fmax(0.0f, out39);
	rsSetElementAt_float(output, out39, base3 + 39 * base4);

	out40 = fmax(0.0f, out40);
	rsSetElementAt_float(output, out40, base3 + 40 * base4);

	out41 = fmax(0.0f, out41);
	rsSetElementAt_float(output, out41, base3 + 41 * base4);

	out42 = fmax(0.0f, out42);
	rsSetElementAt_float(output, out42, base3 + 42 * base4);

	out43 = fmax(0.0f, out43);
	rsSetElementAt_float(output, out43, base3 + 43 * base4);

	out44 = fmax(0.0f, out44);
	rsSetElementAt_float(output, out44, base3 + 44 * base4);

	out45 = fmax(0.0f, out45);
	rsSetElementAt_float(output, out45, base3 + 45 * base4);

	out46 = fmax(0.0f, out46);
	rsSetElementAt_float(output, out46, base3 + 46 * base4);

	out47 = fmax(0.0f, out47);
	rsSetElementAt_float(output, out47, base3 + 47 * base4);

	out48 = fmax(0.0f, out48);
	rsSetElementAt_float(output, out48, base3 + 48 * base4);

	out49 = fmax(0.0f, out49);
	rsSetElementAt_float(output, out49, base3 + 49 * base4);

	out50 = fmax(0.0f, out50);
	rsSetElementAt_float(output, out50, base3 + 50 * base4);

	out51 = fmax(0.0f, out51);
	rsSetElementAt_float(output, out51, base3 + 51 * base4);

}
float __attribute__((kernel)) conv_56(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);
	float out26 = rsGetElementAt_float(bias, m + 26 * parallelOFMs);
	float out27 = rsGetElementAt_float(bias, m + 27 * parallelOFMs);
	float out28 = rsGetElementAt_float(bias, m + 28 * parallelOFMs);
	float out29 = rsGetElementAt_float(bias, m + 29 * parallelOFMs);
	float out30 = rsGetElementAt_float(bias, m + 30 * parallelOFMs);
	float out31 = rsGetElementAt_float(bias, m + 31 * parallelOFMs);
	float out32 = rsGetElementAt_float(bias, m + 32 * parallelOFMs);
	float out33 = rsGetElementAt_float(bias, m + 33 * parallelOFMs);
	float out34 = rsGetElementAt_float(bias, m + 34 * parallelOFMs);
	float out35 = rsGetElementAt_float(bias, m + 35 * parallelOFMs);
	float out36 = rsGetElementAt_float(bias, m + 36 * parallelOFMs);
	float out37 = rsGetElementAt_float(bias, m + 37 * parallelOFMs);
	float out38 = rsGetElementAt_float(bias, m + 38 * parallelOFMs);
	float out39 = rsGetElementAt_float(bias, m + 39 * parallelOFMs);
	float out40 = rsGetElementAt_float(bias, m + 40 * parallelOFMs);
	float out41 = rsGetElementAt_float(bias, m + 41 * parallelOFMs);
	float out42 = rsGetElementAt_float(bias, m + 42 * parallelOFMs);
	float out43 = rsGetElementAt_float(bias, m + 43 * parallelOFMs);
	float out44 = rsGetElementAt_float(bias, m + 44 * parallelOFMs);
	float out45 = rsGetElementAt_float(bias, m + 45 * parallelOFMs);
	float out46 = rsGetElementAt_float(bias, m + 46 * parallelOFMs);
	float out47 = rsGetElementAt_float(bias, m + 47 * parallelOFMs);
	float out48 = rsGetElementAt_float(bias, m + 48 * parallelOFMs);
	float out49 = rsGetElementAt_float(bias, m + 49 * parallelOFMs);
	float out50 = rsGetElementAt_float(bias, m + 50 * parallelOFMs);
	float out51 = rsGetElementAt_float(bias, m + 51 * parallelOFMs);
	float out52 = rsGetElementAt_float(bias, m + 52 * parallelOFMs);
	float out53 = rsGetElementAt_float(bias, m + 53 * parallelOFMs);
	float out54 = rsGetElementAt_float(bias, m + 54 * parallelOFMs);
	float out55 = rsGetElementAt_float(bias, m + 55 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

				float4 wght26 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 26 * parallelOFMs)));
				out26 += dot(ifm, wght26);

				float4 wght27 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 27 * parallelOFMs)));
				out27 += dot(ifm, wght27);

				float4 wght28 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 28 * parallelOFMs)));
				out28 += dot(ifm, wght28);

				float4 wght29 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 29 * parallelOFMs)));
				out29 += dot(ifm, wght29);

				float4 wght30 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 30 * parallelOFMs)));
				out30 += dot(ifm, wght30);

				float4 wght31 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 31 * parallelOFMs)));
				out31 += dot(ifm, wght31);

				float4 wght32 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 32 * parallelOFMs)));
				out32 += dot(ifm, wght32);

				float4 wght33 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 33 * parallelOFMs)));
				out33 += dot(ifm, wght33);

				float4 wght34 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 34 * parallelOFMs)));
				out34 += dot(ifm, wght34);

				float4 wght35 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 35 * parallelOFMs)));
				out35 += dot(ifm, wght35);

				float4 wght36 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 36 * parallelOFMs)));
				out36 += dot(ifm, wght36);

				float4 wght37 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 37 * parallelOFMs)));
				out37 += dot(ifm, wght37);

				float4 wght38 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 38 * parallelOFMs)));
				out38 += dot(ifm, wght38);

				float4 wght39 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 39 * parallelOFMs)));
				out39 += dot(ifm, wght39);

				float4 wght40 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 40 * parallelOFMs)));
				out40 += dot(ifm, wght40);

				float4 wght41 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 41 * parallelOFMs)));
				out41 += dot(ifm, wght41);

				float4 wght42 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 42 * parallelOFMs)));
				out42 += dot(ifm, wght42);

				float4 wght43 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 43 * parallelOFMs)));
				out43 += dot(ifm, wght43);

				float4 wght44 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 44 * parallelOFMs)));
				out44 += dot(ifm, wght44);

				float4 wght45 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 45 * parallelOFMs)));
				out45 += dot(ifm, wght45);

				float4 wght46 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 46 * parallelOFMs)));
				out46 += dot(ifm, wght46);

				float4 wght47 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 47 * parallelOFMs)));
				out47 += dot(ifm, wght47);

				float4 wght48 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 48 * parallelOFMs)));
				out48 += dot(ifm, wght48);

				float4 wght49 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 49 * parallelOFMs)));
				out49 += dot(ifm, wght49);

				float4 wght50 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 50 * parallelOFMs)));
				out50 += dot(ifm, wght50);

				float4 wght51 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 51 * parallelOFMs)));
				out51 += dot(ifm, wght51);

				float4 wght52 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 52 * parallelOFMs)));
				out52 += dot(ifm, wght52);

				float4 wght53 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 53 * parallelOFMs)));
				out53 += dot(ifm, wght53);

				float4 wght54 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 54 * parallelOFMs)));
				out54 += dot(ifm, wght54);

				float4 wght55 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 55 * parallelOFMs)));
				out55 += dot(ifm, wght55);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

	out26 = fmax(0.0f, out26);
	rsSetElementAt_float(output, out26, base3 + 26 * base4);

	out27 = fmax(0.0f, out27);
	rsSetElementAt_float(output, out27, base3 + 27 * base4);

	out28 = fmax(0.0f, out28);
	rsSetElementAt_float(output, out28, base3 + 28 * base4);

	out29 = fmax(0.0f, out29);
	rsSetElementAt_float(output, out29, base3 + 29 * base4);

	out30 = fmax(0.0f, out30);
	rsSetElementAt_float(output, out30, base3 + 30 * base4);

	out31 = fmax(0.0f, out31);
	rsSetElementAt_float(output, out31, base3 + 31 * base4);

	out32 = fmax(0.0f, out32);
	rsSetElementAt_float(output, out32, base3 + 32 * base4);

	out33 = fmax(0.0f, out33);
	rsSetElementAt_float(output, out33, base3 + 33 * base4);

	out34 = fmax(0.0f, out34);
	rsSetElementAt_float(output, out34, base3 + 34 * base4);

	out35 = fmax(0.0f, out35);
	rsSetElementAt_float(output, out35, base3 + 35 * base4);

	out36 = fmax(0.0f, out36);
	rsSetElementAt_float(output, out36, base3 + 36 * base4);

	out37 = fmax(0.0f, out37);
	rsSetElementAt_float(output, out37, base3 + 37 * base4);

	out38 = fmax(0.0f, out38);
	rsSetElementAt_float(output, out38, base3 + 38 * base4);

	out39 = fmax(0.0f, out39);
	rsSetElementAt_float(output, out39, base3 + 39 * base4);

	out40 = fmax(0.0f, out40);
	rsSetElementAt_float(output, out40, base3 + 40 * base4);

	out41 = fmax(0.0f, out41);
	rsSetElementAt_float(output, out41, base3 + 41 * base4);

	out42 = fmax(0.0f, out42);
	rsSetElementAt_float(output, out42, base3 + 42 * base4);

	out43 = fmax(0.0f, out43);
	rsSetElementAt_float(output, out43, base3 + 43 * base4);

	out44 = fmax(0.0f, out44);
	rsSetElementAt_float(output, out44, base3 + 44 * base4);

	out45 = fmax(0.0f, out45);
	rsSetElementAt_float(output, out45, base3 + 45 * base4);

	out46 = fmax(0.0f, out46);
	rsSetElementAt_float(output, out46, base3 + 46 * base4);

	out47 = fmax(0.0f, out47);
	rsSetElementAt_float(output, out47, base3 + 47 * base4);

	out48 = fmax(0.0f, out48);
	rsSetElementAt_float(output, out48, base3 + 48 * base4);

	out49 = fmax(0.0f, out49);
	rsSetElementAt_float(output, out49, base3 + 49 * base4);

	out50 = fmax(0.0f, out50);
	rsSetElementAt_float(output, out50, base3 + 50 * base4);

	out51 = fmax(0.0f, out51);
	rsSetElementAt_float(output, out51, base3 + 51 * base4);

	out52 = fmax(0.0f, out52);
	rsSetElementAt_float(output, out52, base3 + 52 * base4);

	out53 = fmax(0.0f, out53);
	rsSetElementAt_float(output, out53, base3 + 53 * base4);

	out54 = fmax(0.0f, out54);
	rsSetElementAt_float(output, out54, base3 + 54 * base4);

	out55 = fmax(0.0f, out55);
	rsSetElementAt_float(output, out55, base3 + 55 * base4);

}

float __attribute__((kernel)) conv_64(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);
	float out26 = rsGetElementAt_float(bias, m + 26 * parallelOFMs);
	float out27 = rsGetElementAt_float(bias, m + 27 * parallelOFMs);
	float out28 = rsGetElementAt_float(bias, m + 28 * parallelOFMs);
	float out29 = rsGetElementAt_float(bias, m + 29 * parallelOFMs);
	float out30 = rsGetElementAt_float(bias, m + 30 * parallelOFMs);
	float out31 = rsGetElementAt_float(bias, m + 31 * parallelOFMs);
	float out32 = rsGetElementAt_float(bias, m + 32 * parallelOFMs);
	float out33 = rsGetElementAt_float(bias, m + 33 * parallelOFMs);
	float out34 = rsGetElementAt_float(bias, m + 34 * parallelOFMs);
	float out35 = rsGetElementAt_float(bias, m + 35 * parallelOFMs);
	float out36 = rsGetElementAt_float(bias, m + 36 * parallelOFMs);
	float out37 = rsGetElementAt_float(bias, m + 37 * parallelOFMs);
	float out38 = rsGetElementAt_float(bias, m + 38 * parallelOFMs);
	float out39 = rsGetElementAt_float(bias, m + 39 * parallelOFMs);
	float out40 = rsGetElementAt_float(bias, m + 40 * parallelOFMs);
	float out41 = rsGetElementAt_float(bias, m + 41 * parallelOFMs);
	float out42 = rsGetElementAt_float(bias, m + 42 * parallelOFMs);
	float out43 = rsGetElementAt_float(bias, m + 43 * parallelOFMs);
	float out44 = rsGetElementAt_float(bias, m + 44 * parallelOFMs);
	float out45 = rsGetElementAt_float(bias, m + 45 * parallelOFMs);
	float out46 = rsGetElementAt_float(bias, m + 46 * parallelOFMs);
	float out47 = rsGetElementAt_float(bias, m + 47 * parallelOFMs);
	float out48 = rsGetElementAt_float(bias, m + 48 * parallelOFMs);
	float out49 = rsGetElementAt_float(bias, m + 49 * parallelOFMs);
	float out50 = rsGetElementAt_float(bias, m + 50 * parallelOFMs);
	float out51 = rsGetElementAt_float(bias, m + 51 * parallelOFMs);
	float out52 = rsGetElementAt_float(bias, m + 52 * parallelOFMs);
	float out53 = rsGetElementAt_float(bias, m + 53 * parallelOFMs);
	float out54 = rsGetElementAt_float(bias, m + 54 * parallelOFMs);
	float out55 = rsGetElementAt_float(bias, m + 55 * parallelOFMs);
	float out56 = rsGetElementAt_float(bias, m + 56 * parallelOFMs);
	float out57 = rsGetElementAt_float(bias, m + 57 * parallelOFMs);
	float out58 = rsGetElementAt_float(bias, m + 58 * parallelOFMs);
	float out59 = rsGetElementAt_float(bias, m + 59 * parallelOFMs);
	float out60 = rsGetElementAt_float(bias, m + 60 * parallelOFMs);
	float out61 = rsGetElementAt_float(bias, m + 61 * parallelOFMs);
	float out62 = rsGetElementAt_float(bias, m + 62 * parallelOFMs);
	float out63 = rsGetElementAt_float(bias, m + 63 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

				float4 wght26 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 26 * parallelOFMs)));
				out26 += dot(ifm, wght26);

				float4 wght27 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 27 * parallelOFMs)));
				out27 += dot(ifm, wght27);

				float4 wght28 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 28 * parallelOFMs)));
				out28 += dot(ifm, wght28);

				float4 wght29 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 29 * parallelOFMs)));
				out29 += dot(ifm, wght29);

				float4 wght30 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 30 * parallelOFMs)));
				out30 += dot(ifm, wght30);

				float4 wght31 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 31 * parallelOFMs)));
				out31 += dot(ifm, wght31);

				float4 wght32 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 32 * parallelOFMs)));
				out32 += dot(ifm, wght32);

				float4 wght33 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 33 * parallelOFMs)));
				out33 += dot(ifm, wght33);

				float4 wght34 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 34 * parallelOFMs)));
				out34 += dot(ifm, wght34);

				float4 wght35 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 35 * parallelOFMs)));
				out35 += dot(ifm, wght35);

				float4 wght36 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 36 * parallelOFMs)));
				out36 += dot(ifm, wght36);

				float4 wght37 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 37 * parallelOFMs)));
				out37 += dot(ifm, wght37);

				float4 wght38 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 38 * parallelOFMs)));
				out38 += dot(ifm, wght38);

				float4 wght39 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 39 * parallelOFMs)));
				out39 += dot(ifm, wght39);

				float4 wght40 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 40 * parallelOFMs)));
				out40 += dot(ifm, wght40);

				float4 wght41 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 41 * parallelOFMs)));
				out41 += dot(ifm, wght41);

				float4 wght42 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 42 * parallelOFMs)));
				out42 += dot(ifm, wght42);

				float4 wght43 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 43 * parallelOFMs)));
				out43 += dot(ifm, wght43);

				float4 wght44 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 44 * parallelOFMs)));
				out44 += dot(ifm, wght44);

				float4 wght45 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 45 * parallelOFMs)));
				out45 += dot(ifm, wght45);

				float4 wght46 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 46 * parallelOFMs)));
				out46 += dot(ifm, wght46);

				float4 wght47 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 47 * parallelOFMs)));
				out47 += dot(ifm, wght47);

				float4 wght48 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 48 * parallelOFMs)));
				out48 += dot(ifm, wght48);

				float4 wght49 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 49 * parallelOFMs)));
				out49 += dot(ifm, wght49);

				float4 wght50 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 50 * parallelOFMs)));
				out50 += dot(ifm, wght50);

				float4 wght51 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 51 * parallelOFMs)));
				out51 += dot(ifm, wght51);

				float4 wght52 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 52 * parallelOFMs)));
				out52 += dot(ifm, wght52);

				float4 wght53 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 53 * parallelOFMs)));
				out53 += dot(ifm, wght53);

				float4 wght54 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 54 * parallelOFMs)));
				out54 += dot(ifm, wght54);

				float4 wght55 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 55 * parallelOFMs)));
				out55 += dot(ifm, wght55);

				float4 wght56 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 56 * parallelOFMs)));
				out56 += dot(ifm, wght56);

				float4 wght57 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 57 * parallelOFMs)));
				out57 += dot(ifm, wght57);

				float4 wght58 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 58 * parallelOFMs)));
				out58 += dot(ifm, wght58);

				float4 wght59 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 59 * parallelOFMs)));
				out59 += dot(ifm, wght59);

				float4 wght60 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 60 * parallelOFMs)));
				out60 += dot(ifm, wght60);

				float4 wght61 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 61 * parallelOFMs)));
				out61 += dot(ifm, wght61);

				float4 wght62 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 62 * parallelOFMs)));
				out62 += dot(ifm, wght62);

				float4 wght63 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 63 * parallelOFMs)));
				out63 += dot(ifm, wght63);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

	out26 = fmax(0.0f, out26);
	rsSetElementAt_float(output, out26, base3 + 26 * base4);

	out27 = fmax(0.0f, out27);
	rsSetElementAt_float(output, out27, base3 + 27 * base4);

	out28 = fmax(0.0f, out28);
	rsSetElementAt_float(output, out28, base3 + 28 * base4);

	out29 = fmax(0.0f, out29);
	rsSetElementAt_float(output, out29, base3 + 29 * base4);

	out30 = fmax(0.0f, out30);
	rsSetElementAt_float(output, out30, base3 + 30 * base4);

	out31 = fmax(0.0f, out31);
	rsSetElementAt_float(output, out31, base3 + 31 * base4);

	out32 = fmax(0.0f, out32);
	rsSetElementAt_float(output, out32, base3 + 32 * base4);

	out33 = fmax(0.0f, out33);
	rsSetElementAt_float(output, out33, base3 + 33 * base4);

	out34 = fmax(0.0f, out34);
	rsSetElementAt_float(output, out34, base3 + 34 * base4);

	out35 = fmax(0.0f, out35);
	rsSetElementAt_float(output, out35, base3 + 35 * base4);

	out36 = fmax(0.0f, out36);
	rsSetElementAt_float(output, out36, base3 + 36 * base4);

	out37 = fmax(0.0f, out37);
	rsSetElementAt_float(output, out37, base3 + 37 * base4);

	out38 = fmax(0.0f, out38);
	rsSetElementAt_float(output, out38, base3 + 38 * base4);

	out39 = fmax(0.0f, out39);
	rsSetElementAt_float(output, out39, base3 + 39 * base4);

	out40 = fmax(0.0f, out40);
	rsSetElementAt_float(output, out40, base3 + 40 * base4);

	out41 = fmax(0.0f, out41);
	rsSetElementAt_float(output, out41, base3 + 41 * base4);

	out42 = fmax(0.0f, out42);
	rsSetElementAt_float(output, out42, base3 + 42 * base4);

	out43 = fmax(0.0f, out43);
	rsSetElementAt_float(output, out43, base3 + 43 * base4);

	out44 = fmax(0.0f, out44);
	rsSetElementAt_float(output, out44, base3 + 44 * base4);

	out45 = fmax(0.0f, out45);
	rsSetElementAt_float(output, out45, base3 + 45 * base4);

	out46 = fmax(0.0f, out46);
	rsSetElementAt_float(output, out46, base3 + 46 * base4);

	out47 = fmax(0.0f, out47);
	rsSetElementAt_float(output, out47, base3 + 47 * base4);

	out48 = fmax(0.0f, out48);
	rsSetElementAt_float(output, out48, base3 + 48 * base4);

	out49 = fmax(0.0f, out49);
	rsSetElementAt_float(output, out49, base3 + 49 * base4);

	out50 = fmax(0.0f, out50);
	rsSetElementAt_float(output, out50, base3 + 50 * base4);

	out51 = fmax(0.0f, out51);
	rsSetElementAt_float(output, out51, base3 + 51 * base4);

	out52 = fmax(0.0f, out52);
	rsSetElementAt_float(output, out52, base3 + 52 * base4);

	out53 = fmax(0.0f, out53);
	rsSetElementAt_float(output, out53, base3 + 53 * base4);

	out54 = fmax(0.0f, out54);
	rsSetElementAt_float(output, out54, base3 + 54 * base4);

	out55 = fmax(0.0f, out55);
	rsSetElementAt_float(output, out55, base3 + 55 * base4);

	out56 = fmax(0.0f, out56);
	rsSetElementAt_float(output, out56, base3 + 56 * base4);

	out57 = fmax(0.0f, out57);
	rsSetElementAt_float(output, out57, base3 + 57 * base4);

	out58 = fmax(0.0f, out58);
	rsSetElementAt_float(output, out58, base3 + 58 * base4);

	out59 = fmax(0.0f, out59);
	rsSetElementAt_float(output, out59, base3 + 59 * base4);

	out60 = fmax(0.0f, out60);
	rsSetElementAt_float(output, out60, base3 + 60 * base4);

	out61 = fmax(0.0f, out61);
	rsSetElementAt_float(output, out61, base3 + 61 * base4);

	out62 = fmax(0.0f, out62);
	rsSetElementAt_float(output, out62, base3 + 62 * base4);

	out63 = fmax(0.0f, out63);
	rsSetElementAt_float(output, out63, base3 + 63 * base4);

}
float __attribute__((kernel)) conv_72(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);
	float out26 = rsGetElementAt_float(bias, m + 26 * parallelOFMs);
	float out27 = rsGetElementAt_float(bias, m + 27 * parallelOFMs);
	float out28 = rsGetElementAt_float(bias, m + 28 * parallelOFMs);
	float out29 = rsGetElementAt_float(bias, m + 29 * parallelOFMs);
	float out30 = rsGetElementAt_float(bias, m + 30 * parallelOFMs);
	float out31 = rsGetElementAt_float(bias, m + 31 * parallelOFMs);
	float out32 = rsGetElementAt_float(bias, m + 32 * parallelOFMs);
	float out33 = rsGetElementAt_float(bias, m + 33 * parallelOFMs);
	float out34 = rsGetElementAt_float(bias, m + 34 * parallelOFMs);
	float out35 = rsGetElementAt_float(bias, m + 35 * parallelOFMs);
	float out36 = rsGetElementAt_float(bias, m + 36 * parallelOFMs);
	float out37 = rsGetElementAt_float(bias, m + 37 * parallelOFMs);
	float out38 = rsGetElementAt_float(bias, m + 38 * parallelOFMs);
	float out39 = rsGetElementAt_float(bias, m + 39 * parallelOFMs);
	float out40 = rsGetElementAt_float(bias, m + 40 * parallelOFMs);
	float out41 = rsGetElementAt_float(bias, m + 41 * parallelOFMs);
	float out42 = rsGetElementAt_float(bias, m + 42 * parallelOFMs);
	float out43 = rsGetElementAt_float(bias, m + 43 * parallelOFMs);
	float out44 = rsGetElementAt_float(bias, m + 44 * parallelOFMs);
	float out45 = rsGetElementAt_float(bias, m + 45 * parallelOFMs);
	float out46 = rsGetElementAt_float(bias, m + 46 * parallelOFMs);
	float out47 = rsGetElementAt_float(bias, m + 47 * parallelOFMs);
	float out48 = rsGetElementAt_float(bias, m + 48 * parallelOFMs);
	float out49 = rsGetElementAt_float(bias, m + 49 * parallelOFMs);
	float out50 = rsGetElementAt_float(bias, m + 50 * parallelOFMs);
	float out51 = rsGetElementAt_float(bias, m + 51 * parallelOFMs);
	float out52 = rsGetElementAt_float(bias, m + 52 * parallelOFMs);
	float out53 = rsGetElementAt_float(bias, m + 53 * parallelOFMs);
	float out54 = rsGetElementAt_float(bias, m + 54 * parallelOFMs);
	float out55 = rsGetElementAt_float(bias, m + 55 * parallelOFMs);
	float out56 = rsGetElementAt_float(bias, m + 56 * parallelOFMs);
	float out57 = rsGetElementAt_float(bias, m + 57 * parallelOFMs);
	float out58 = rsGetElementAt_float(bias, m + 58 * parallelOFMs);
	float out59 = rsGetElementAt_float(bias, m + 59 * parallelOFMs);
	float out60 = rsGetElementAt_float(bias, m + 60 * parallelOFMs);
	float out61 = rsGetElementAt_float(bias, m + 61 * parallelOFMs);
	float out62 = rsGetElementAt_float(bias, m + 62 * parallelOFMs);
	float out63 = rsGetElementAt_float(bias, m + 63 * parallelOFMs);
	float out64 = rsGetElementAt_float(bias, m + 64 * parallelOFMs);
	float out65 = rsGetElementAt_float(bias, m + 65 * parallelOFMs);
	float out66 = rsGetElementAt_float(bias, m + 66 * parallelOFMs);
	float out67 = rsGetElementAt_float(bias, m + 67 * parallelOFMs);
	float out68 = rsGetElementAt_float(bias, m + 68 * parallelOFMs);
	float out69 = rsGetElementAt_float(bias, m + 69 * parallelOFMs);
	float out70 = rsGetElementAt_float(bias, m + 70 * parallelOFMs);
	float out71 = rsGetElementAt_float(bias, m + 71 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

				float4 wght26 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 26 * parallelOFMs)));
				out26 += dot(ifm, wght26);

				float4 wght27 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 27 * parallelOFMs)));
				out27 += dot(ifm, wght27);

				float4 wght28 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 28 * parallelOFMs)));
				out28 += dot(ifm, wght28);

				float4 wght29 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 29 * parallelOFMs)));
				out29 += dot(ifm, wght29);

				float4 wght30 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 30 * parallelOFMs)));
				out30 += dot(ifm, wght30);

				float4 wght31 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 31 * parallelOFMs)));
				out31 += dot(ifm, wght31);

				float4 wght32 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 32 * parallelOFMs)));
				out32 += dot(ifm, wght32);

				float4 wght33 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 33 * parallelOFMs)));
				out33 += dot(ifm, wght33);

				float4 wght34 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 34 * parallelOFMs)));
				out34 += dot(ifm, wght34);

				float4 wght35 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 35 * parallelOFMs)));
				out35 += dot(ifm, wght35);

				float4 wght36 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 36 * parallelOFMs)));
				out36 += dot(ifm, wght36);

				float4 wght37 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 37 * parallelOFMs)));
				out37 += dot(ifm, wght37);

				float4 wght38 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 38 * parallelOFMs)));
				out38 += dot(ifm, wght38);

				float4 wght39 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 39 * parallelOFMs)));
				out39 += dot(ifm, wght39);

				float4 wght40 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 40 * parallelOFMs)));
				out40 += dot(ifm, wght40);

				float4 wght41 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 41 * parallelOFMs)));
				out41 += dot(ifm, wght41);

				float4 wght42 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 42 * parallelOFMs)));
				out42 += dot(ifm, wght42);

				float4 wght43 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 43 * parallelOFMs)));
				out43 += dot(ifm, wght43);

				float4 wght44 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 44 * parallelOFMs)));
				out44 += dot(ifm, wght44);

				float4 wght45 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 45 * parallelOFMs)));
				out45 += dot(ifm, wght45);

				float4 wght46 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 46 * parallelOFMs)));
				out46 += dot(ifm, wght46);

				float4 wght47 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 47 * parallelOFMs)));
				out47 += dot(ifm, wght47);

				float4 wght48 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 48 * parallelOFMs)));
				out48 += dot(ifm, wght48);

				float4 wght49 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 49 * parallelOFMs)));
				out49 += dot(ifm, wght49);

				float4 wght50 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 50 * parallelOFMs)));
				out50 += dot(ifm, wght50);

				float4 wght51 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 51 * parallelOFMs)));
				out51 += dot(ifm, wght51);

				float4 wght52 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 52 * parallelOFMs)));
				out52 += dot(ifm, wght52);

				float4 wght53 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 53 * parallelOFMs)));
				out53 += dot(ifm, wght53);

				float4 wght54 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 54 * parallelOFMs)));
				out54 += dot(ifm, wght54);

				float4 wght55 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 55 * parallelOFMs)));
				out55 += dot(ifm, wght55);

				float4 wght56 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 56 * parallelOFMs)));
				out56 += dot(ifm, wght56);

				float4 wght57 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 57 * parallelOFMs)));
				out57 += dot(ifm, wght57);

				float4 wght58 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 58 * parallelOFMs)));
				out58 += dot(ifm, wght58);

				float4 wght59 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 59 * parallelOFMs)));
				out59 += dot(ifm, wght59);

				float4 wght60 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 60 * parallelOFMs)));
				out60 += dot(ifm, wght60);

				float4 wght61 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 61 * parallelOFMs)));
				out61 += dot(ifm, wght61);

				float4 wght62 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 62 * parallelOFMs)));
				out62 += dot(ifm, wght62);

				float4 wght63 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 63 * parallelOFMs)));
				out63 += dot(ifm, wght63);

				float4 wght64 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 64 * parallelOFMs)));
				out64 += dot(ifm, wght64);

				float4 wght65 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 65 * parallelOFMs)));
				out65 += dot(ifm, wght65);

				float4 wght66 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 66 * parallelOFMs)));
				out66 += dot(ifm, wght66);

				float4 wght67 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 67 * parallelOFMs)));
				out67 += dot(ifm, wght67);

				float4 wght68 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 68 * parallelOFMs)));
				out68 += dot(ifm, wght68);

				float4 wght69 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 69 * parallelOFMs)));
				out69 += dot(ifm, wght69);

				float4 wght70 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 70 * parallelOFMs)));
				out70 += dot(ifm, wght70);

				float4 wght71 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 71 * parallelOFMs)));
				out71 += dot(ifm, wght71);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

	out26 = fmax(0.0f, out26);
	rsSetElementAt_float(output, out26, base3 + 26 * base4);

	out27 = fmax(0.0f, out27);
	rsSetElementAt_float(output, out27, base3 + 27 * base4);

	out28 = fmax(0.0f, out28);
	rsSetElementAt_float(output, out28, base3 + 28 * base4);

	out29 = fmax(0.0f, out29);
	rsSetElementAt_float(output, out29, base3 + 29 * base4);

	out30 = fmax(0.0f, out30);
	rsSetElementAt_float(output, out30, base3 + 30 * base4);

	out31 = fmax(0.0f, out31);
	rsSetElementAt_float(output, out31, base3 + 31 * base4);

	out32 = fmax(0.0f, out32);
	rsSetElementAt_float(output, out32, base3 + 32 * base4);

	out33 = fmax(0.0f, out33);
	rsSetElementAt_float(output, out33, base3 + 33 * base4);

	out34 = fmax(0.0f, out34);
	rsSetElementAt_float(output, out34, base3 + 34 * base4);

	out35 = fmax(0.0f, out35);
	rsSetElementAt_float(output, out35, base3 + 35 * base4);

	out36 = fmax(0.0f, out36);
	rsSetElementAt_float(output, out36, base3 + 36 * base4);

	out37 = fmax(0.0f, out37);
	rsSetElementAt_float(output, out37, base3 + 37 * base4);

	out38 = fmax(0.0f, out38);
	rsSetElementAt_float(output, out38, base3 + 38 * base4);

	out39 = fmax(0.0f, out39);
	rsSetElementAt_float(output, out39, base3 + 39 * base4);

	out40 = fmax(0.0f, out40);
	rsSetElementAt_float(output, out40, base3 + 40 * base4);

	out41 = fmax(0.0f, out41);
	rsSetElementAt_float(output, out41, base3 + 41 * base4);

	out42 = fmax(0.0f, out42);
	rsSetElementAt_float(output, out42, base3 + 42 * base4);

	out43 = fmax(0.0f, out43);
	rsSetElementAt_float(output, out43, base3 + 43 * base4);

	out44 = fmax(0.0f, out44);
	rsSetElementAt_float(output, out44, base3 + 44 * base4);

	out45 = fmax(0.0f, out45);
	rsSetElementAt_float(output, out45, base3 + 45 * base4);

	out46 = fmax(0.0f, out46);
	rsSetElementAt_float(output, out46, base3 + 46 * base4);

	out47 = fmax(0.0f, out47);
	rsSetElementAt_float(output, out47, base3 + 47 * base4);

	out48 = fmax(0.0f, out48);
	rsSetElementAt_float(output, out48, base3 + 48 * base4);

	out49 = fmax(0.0f, out49);
	rsSetElementAt_float(output, out49, base3 + 49 * base4);

	out50 = fmax(0.0f, out50);
	rsSetElementAt_float(output, out50, base3 + 50 * base4);

	out51 = fmax(0.0f, out51);
	rsSetElementAt_float(output, out51, base3 + 51 * base4);

	out52 = fmax(0.0f, out52);
	rsSetElementAt_float(output, out52, base3 + 52 * base4);

	out53 = fmax(0.0f, out53);
	rsSetElementAt_float(output, out53, base3 + 53 * base4);

	out54 = fmax(0.0f, out54);
	rsSetElementAt_float(output, out54, base3 + 54 * base4);

	out55 = fmax(0.0f, out55);
	rsSetElementAt_float(output, out55, base3 + 55 * base4);

	out56 = fmax(0.0f, out56);
	rsSetElementAt_float(output, out56, base3 + 56 * base4);

	out57 = fmax(0.0f, out57);
	rsSetElementAt_float(output, out57, base3 + 57 * base4);

	out58 = fmax(0.0f, out58);
	rsSetElementAt_float(output, out58, base3 + 58 * base4);

	out59 = fmax(0.0f, out59);
	rsSetElementAt_float(output, out59, base3 + 59 * base4);

	out60 = fmax(0.0f, out60);
	rsSetElementAt_float(output, out60, base3 + 60 * base4);

	out61 = fmax(0.0f, out61);
	rsSetElementAt_float(output, out61, base3 + 61 * base4);

	out62 = fmax(0.0f, out62);
	rsSetElementAt_float(output, out62, base3 + 62 * base4);

	out63 = fmax(0.0f, out63);
	rsSetElementAt_float(output, out63, base3 + 63 * base4);

	out64 = fmax(0.0f, out64);
	rsSetElementAt_float(output, out64, base3 + 64 * base4);

	out65 = fmax(0.0f, out65);
	rsSetElementAt_float(output, out65, base3 + 65 * base4);

	out66 = fmax(0.0f, out66);
	rsSetElementAt_float(output, out66, base3 + 66 * base4);

	out67 = fmax(0.0f, out67);
	rsSetElementAt_float(output, out67, base3 + 67 * base4);

	out68 = fmax(0.0f, out68);
	rsSetElementAt_float(output, out68, base3 + 68 * base4);

	out69 = fmax(0.0f, out69);
	rsSetElementAt_float(output, out69, base3 + 69 * base4);

	out70 = fmax(0.0f, out70);
	rsSetElementAt_float(output, out70, base3 + 70 * base4);

	out71 = fmax(0.0f, out71);
	rsSetElementAt_float(output, out71, base3 + 71 * base4);

}
float __attribute__((kernel)) conv_80(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);
	float out26 = rsGetElementAt_float(bias, m + 26 * parallelOFMs);
	float out27 = rsGetElementAt_float(bias, m + 27 * parallelOFMs);
	float out28 = rsGetElementAt_float(bias, m + 28 * parallelOFMs);
	float out29 = rsGetElementAt_float(bias, m + 29 * parallelOFMs);
	float out30 = rsGetElementAt_float(bias, m + 30 * parallelOFMs);
	float out31 = rsGetElementAt_float(bias, m + 31 * parallelOFMs);
	float out32 = rsGetElementAt_float(bias, m + 32 * parallelOFMs);
	float out33 = rsGetElementAt_float(bias, m + 33 * parallelOFMs);
	float out34 = rsGetElementAt_float(bias, m + 34 * parallelOFMs);
	float out35 = rsGetElementAt_float(bias, m + 35 * parallelOFMs);
	float out36 = rsGetElementAt_float(bias, m + 36 * parallelOFMs);
	float out37 = rsGetElementAt_float(bias, m + 37 * parallelOFMs);
	float out38 = rsGetElementAt_float(bias, m + 38 * parallelOFMs);
	float out39 = rsGetElementAt_float(bias, m + 39 * parallelOFMs);
	float out40 = rsGetElementAt_float(bias, m + 40 * parallelOFMs);
	float out41 = rsGetElementAt_float(bias, m + 41 * parallelOFMs);
	float out42 = rsGetElementAt_float(bias, m + 42 * parallelOFMs);
	float out43 = rsGetElementAt_float(bias, m + 43 * parallelOFMs);
	float out44 = rsGetElementAt_float(bias, m + 44 * parallelOFMs);
	float out45 = rsGetElementAt_float(bias, m + 45 * parallelOFMs);
	float out46 = rsGetElementAt_float(bias, m + 46 * parallelOFMs);
	float out47 = rsGetElementAt_float(bias, m + 47 * parallelOFMs);
	float out48 = rsGetElementAt_float(bias, m + 48 * parallelOFMs);
	float out49 = rsGetElementAt_float(bias, m + 49 * parallelOFMs);
	float out50 = rsGetElementAt_float(bias, m + 50 * parallelOFMs);
	float out51 = rsGetElementAt_float(bias, m + 51 * parallelOFMs);
	float out52 = rsGetElementAt_float(bias, m + 52 * parallelOFMs);
	float out53 = rsGetElementAt_float(bias, m + 53 * parallelOFMs);
	float out54 = rsGetElementAt_float(bias, m + 54 * parallelOFMs);
	float out55 = rsGetElementAt_float(bias, m + 55 * parallelOFMs);
	float out56 = rsGetElementAt_float(bias, m + 56 * parallelOFMs);
	float out57 = rsGetElementAt_float(bias, m + 57 * parallelOFMs);
	float out58 = rsGetElementAt_float(bias, m + 58 * parallelOFMs);
	float out59 = rsGetElementAt_float(bias, m + 59 * parallelOFMs);
	float out60 = rsGetElementAt_float(bias, m + 60 * parallelOFMs);
	float out61 = rsGetElementAt_float(bias, m + 61 * parallelOFMs);
	float out62 = rsGetElementAt_float(bias, m + 62 * parallelOFMs);
	float out63 = rsGetElementAt_float(bias, m + 63 * parallelOFMs);
	float out64 = rsGetElementAt_float(bias, m + 64 * parallelOFMs);
	float out65 = rsGetElementAt_float(bias, m + 65 * parallelOFMs);
	float out66 = rsGetElementAt_float(bias, m + 66 * parallelOFMs);
	float out67 = rsGetElementAt_float(bias, m + 67 * parallelOFMs);
	float out68 = rsGetElementAt_float(bias, m + 68 * parallelOFMs);
	float out69 = rsGetElementAt_float(bias, m + 69 * parallelOFMs);
	float out70 = rsGetElementAt_float(bias, m + 70 * parallelOFMs);
	float out71 = rsGetElementAt_float(bias, m + 71 * parallelOFMs);
	float out72 = rsGetElementAt_float(bias, m + 72 * parallelOFMs);
	float out73 = rsGetElementAt_float(bias, m + 73 * parallelOFMs);
	float out74 = rsGetElementAt_float(bias, m + 74 * parallelOFMs);
	float out75 = rsGetElementAt_float(bias, m + 75 * parallelOFMs);
	float out76 = rsGetElementAt_float(bias, m + 76 * parallelOFMs);
	float out77 = rsGetElementAt_float(bias, m + 77 * parallelOFMs);
	float out78 = rsGetElementAt_float(bias, m + 78 * parallelOFMs);
	float out79 = rsGetElementAt_float(bias, m + 79 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

				float4 wght26 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 26 * parallelOFMs)));
				out26 += dot(ifm, wght26);

				float4 wght27 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 27 * parallelOFMs)));
				out27 += dot(ifm, wght27);

				float4 wght28 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 28 * parallelOFMs)));
				out28 += dot(ifm, wght28);

				float4 wght29 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 29 * parallelOFMs)));
				out29 += dot(ifm, wght29);

				float4 wght30 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 30 * parallelOFMs)));
				out30 += dot(ifm, wght30);

				float4 wght31 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 31 * parallelOFMs)));
				out31 += dot(ifm, wght31);

				float4 wght32 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 32 * parallelOFMs)));
				out32 += dot(ifm, wght32);

				float4 wght33 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 33 * parallelOFMs)));
				out33 += dot(ifm, wght33);

				float4 wght34 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 34 * parallelOFMs)));
				out34 += dot(ifm, wght34);

				float4 wght35 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 35 * parallelOFMs)));
				out35 += dot(ifm, wght35);

				float4 wght36 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 36 * parallelOFMs)));
				out36 += dot(ifm, wght36);

				float4 wght37 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 37 * parallelOFMs)));
				out37 += dot(ifm, wght37);

				float4 wght38 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 38 * parallelOFMs)));
				out38 += dot(ifm, wght38);

				float4 wght39 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 39 * parallelOFMs)));
				out39 += dot(ifm, wght39);

				float4 wght40 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 40 * parallelOFMs)));
				out40 += dot(ifm, wght40);

				float4 wght41 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 41 * parallelOFMs)));
				out41 += dot(ifm, wght41);

				float4 wght42 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 42 * parallelOFMs)));
				out42 += dot(ifm, wght42);

				float4 wght43 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 43 * parallelOFMs)));
				out43 += dot(ifm, wght43);

				float4 wght44 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 44 * parallelOFMs)));
				out44 += dot(ifm, wght44);

				float4 wght45 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 45 * parallelOFMs)));
				out45 += dot(ifm, wght45);

				float4 wght46 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 46 * parallelOFMs)));
				out46 += dot(ifm, wght46);

				float4 wght47 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 47 * parallelOFMs)));
				out47 += dot(ifm, wght47);

				float4 wght48 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 48 * parallelOFMs)));
				out48 += dot(ifm, wght48);

				float4 wght49 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 49 * parallelOFMs)));
				out49 += dot(ifm, wght49);

				float4 wght50 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 50 * parallelOFMs)));
				out50 += dot(ifm, wght50);

				float4 wght51 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 51 * parallelOFMs)));
				out51 += dot(ifm, wght51);

				float4 wght52 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 52 * parallelOFMs)));
				out52 += dot(ifm, wght52);

				float4 wght53 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 53 * parallelOFMs)));
				out53 += dot(ifm, wght53);

				float4 wght54 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 54 * parallelOFMs)));
				out54 += dot(ifm, wght54);

				float4 wght55 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 55 * parallelOFMs)));
				out55 += dot(ifm, wght55);

				float4 wght56 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 56 * parallelOFMs)));
				out56 += dot(ifm, wght56);

				float4 wght57 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 57 * parallelOFMs)));
				out57 += dot(ifm, wght57);

				float4 wght58 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 58 * parallelOFMs)));
				out58 += dot(ifm, wght58);

				float4 wght59 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 59 * parallelOFMs)));
				out59 += dot(ifm, wght59);

				float4 wght60 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 60 * parallelOFMs)));
				out60 += dot(ifm, wght60);

				float4 wght61 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 61 * parallelOFMs)));
				out61 += dot(ifm, wght61);

				float4 wght62 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 62 * parallelOFMs)));
				out62 += dot(ifm, wght62);

				float4 wght63 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 63 * parallelOFMs)));
				out63 += dot(ifm, wght63);

				float4 wght64 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 64 * parallelOFMs)));
				out64 += dot(ifm, wght64);

				float4 wght65 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 65 * parallelOFMs)));
				out65 += dot(ifm, wght65);

				float4 wght66 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 66 * parallelOFMs)));
				out66 += dot(ifm, wght66);

				float4 wght67 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 67 * parallelOFMs)));
				out67 += dot(ifm, wght67);

				float4 wght68 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 68 * parallelOFMs)));
				out68 += dot(ifm, wght68);

				float4 wght69 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 69 * parallelOFMs)));
				out69 += dot(ifm, wght69);

				float4 wght70 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 70 * parallelOFMs)));
				out70 += dot(ifm, wght70);

				float4 wght71 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 71 * parallelOFMs)));
				out71 += dot(ifm, wght71);

				float4 wght72 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 72 * parallelOFMs)));
				out72 += dot(ifm, wght72);

				float4 wght73 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 73 * parallelOFMs)));
				out73 += dot(ifm, wght73);

				float4 wght74 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 74 * parallelOFMs)));
				out74 += dot(ifm, wght74);

				float4 wght75 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 75 * parallelOFMs)));
				out75 += dot(ifm, wght75);

				float4 wght76 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 76 * parallelOFMs)));
				out76 += dot(ifm, wght76);

				float4 wght77 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 77 * parallelOFMs)));
				out77 += dot(ifm, wght77);

				float4 wght78 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 78 * parallelOFMs)));
				out78 += dot(ifm, wght78);

				float4 wght79 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 79 * parallelOFMs)));
				out79 += dot(ifm, wght79);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

	out26 = fmax(0.0f, out26);
	rsSetElementAt_float(output, out26, base3 + 26 * base4);

	out27 = fmax(0.0f, out27);
	rsSetElementAt_float(output, out27, base3 + 27 * base4);

	out28 = fmax(0.0f, out28);
	rsSetElementAt_float(output, out28, base3 + 28 * base4);

	out29 = fmax(0.0f, out29);
	rsSetElementAt_float(output, out29, base3 + 29 * base4);

	out30 = fmax(0.0f, out30);
	rsSetElementAt_float(output, out30, base3 + 30 * base4);

	out31 = fmax(0.0f, out31);
	rsSetElementAt_float(output, out31, base3 + 31 * base4);

	out32 = fmax(0.0f, out32);
	rsSetElementAt_float(output, out32, base3 + 32 * base4);

	out33 = fmax(0.0f, out33);
	rsSetElementAt_float(output, out33, base3 + 33 * base4);

	out34 = fmax(0.0f, out34);
	rsSetElementAt_float(output, out34, base3 + 34 * base4);

	out35 = fmax(0.0f, out35);
	rsSetElementAt_float(output, out35, base3 + 35 * base4);

	out36 = fmax(0.0f, out36);
	rsSetElementAt_float(output, out36, base3 + 36 * base4);

	out37 = fmax(0.0f, out37);
	rsSetElementAt_float(output, out37, base3 + 37 * base4);

	out38 = fmax(0.0f, out38);
	rsSetElementAt_float(output, out38, base3 + 38 * base4);

	out39 = fmax(0.0f, out39);
	rsSetElementAt_float(output, out39, base3 + 39 * base4);

	out40 = fmax(0.0f, out40);
	rsSetElementAt_float(output, out40, base3 + 40 * base4);

	out41 = fmax(0.0f, out41);
	rsSetElementAt_float(output, out41, base3 + 41 * base4);

	out42 = fmax(0.0f, out42);
	rsSetElementAt_float(output, out42, base3 + 42 * base4);

	out43 = fmax(0.0f, out43);
	rsSetElementAt_float(output, out43, base3 + 43 * base4);

	out44 = fmax(0.0f, out44);
	rsSetElementAt_float(output, out44, base3 + 44 * base4);

	out45 = fmax(0.0f, out45);
	rsSetElementAt_float(output, out45, base3 + 45 * base4);

	out46 = fmax(0.0f, out46);
	rsSetElementAt_float(output, out46, base3 + 46 * base4);

	out47 = fmax(0.0f, out47);
	rsSetElementAt_float(output, out47, base3 + 47 * base4);

	out48 = fmax(0.0f, out48);
	rsSetElementAt_float(output, out48, base3 + 48 * base4);

	out49 = fmax(0.0f, out49);
	rsSetElementAt_float(output, out49, base3 + 49 * base4);

	out50 = fmax(0.0f, out50);
	rsSetElementAt_float(output, out50, base3 + 50 * base4);

	out51 = fmax(0.0f, out51);
	rsSetElementAt_float(output, out51, base3 + 51 * base4);

	out52 = fmax(0.0f, out52);
	rsSetElementAt_float(output, out52, base3 + 52 * base4);

	out53 = fmax(0.0f, out53);
	rsSetElementAt_float(output, out53, base3 + 53 * base4);

	out54 = fmax(0.0f, out54);
	rsSetElementAt_float(output, out54, base3 + 54 * base4);

	out55 = fmax(0.0f, out55);
	rsSetElementAt_float(output, out55, base3 + 55 * base4);

	out56 = fmax(0.0f, out56);
	rsSetElementAt_float(output, out56, base3 + 56 * base4);

	out57 = fmax(0.0f, out57);
	rsSetElementAt_float(output, out57, base3 + 57 * base4);

	out58 = fmax(0.0f, out58);
	rsSetElementAt_float(output, out58, base3 + 58 * base4);

	out59 = fmax(0.0f, out59);
	rsSetElementAt_float(output, out59, base3 + 59 * base4);

	out60 = fmax(0.0f, out60);
	rsSetElementAt_float(output, out60, base3 + 60 * base4);

	out61 = fmax(0.0f, out61);
	rsSetElementAt_float(output, out61, base3 + 61 * base4);

	out62 = fmax(0.0f, out62);
	rsSetElementAt_float(output, out62, base3 + 62 * base4);

	out63 = fmax(0.0f, out63);
	rsSetElementAt_float(output, out63, base3 + 63 * base4);

	out64 = fmax(0.0f, out64);
	rsSetElementAt_float(output, out64, base3 + 64 * base4);

	out65 = fmax(0.0f, out65);
	rsSetElementAt_float(output, out65, base3 + 65 * base4);

	out66 = fmax(0.0f, out66);
	rsSetElementAt_float(output, out66, base3 + 66 * base4);

	out67 = fmax(0.0f, out67);
	rsSetElementAt_float(output, out67, base3 + 67 * base4);

	out68 = fmax(0.0f, out68);
	rsSetElementAt_float(output, out68, base3 + 68 * base4);

	out69 = fmax(0.0f, out69);
	rsSetElementAt_float(output, out69, base3 + 69 * base4);

	out70 = fmax(0.0f, out70);
	rsSetElementAt_float(output, out70, base3 + 70 * base4);

	out71 = fmax(0.0f, out71);
	rsSetElementAt_float(output, out71, base3 + 71 * base4);

	out72 = fmax(0.0f, out72);
	rsSetElementAt_float(output, out72, base3 + 72 * base4);

	out73 = fmax(0.0f, out73);
	rsSetElementAt_float(output, out73, base3 + 73 * base4);

	out74 = fmax(0.0f, out74);
	rsSetElementAt_float(output, out74, base3 + 74 * base4);

	out75 = fmax(0.0f, out75);
	rsSetElementAt_float(output, out75, base3 + 75 * base4);

	out76 = fmax(0.0f, out76);
	rsSetElementAt_float(output, out76, base3 + 76 * base4);

	out77 = fmax(0.0f, out77);
	rsSetElementAt_float(output, out77, base3 + 77 * base4);

	out78 = fmax(0.0f, out78);
	rsSetElementAt_float(output, out78, base3 + 78 * base4);

	out79 = fmax(0.0f, out79);
	rsSetElementAt_float(output, out79, base3 + 79 * base4);

}
float __attribute__((kernel)) conv_96(uint32_t x) {
	int32_t w = (x / 4) % Wout;
	int32_t h = (((x - 4 * w) / 4) / Wout) % Hout;
	int32_t m = (x % 4) + (x / (4 * Wout * Hout)) * 4;

	int32_t n, i, j;

	float out0 = rsGetElementAt_float(bias, m);
	float out1 = rsGetElementAt_float(bias, m + parallelOFMs);
	float out2 = rsGetElementAt_float(bias, m + 2 * parallelOFMs);
	float out3 = rsGetElementAt_float(bias, m + 3 * parallelOFMs);
	float out4 = rsGetElementAt_float(bias, m + 4 * parallelOFMs);
	float out5 = rsGetElementAt_float(bias, m + 5 * parallelOFMs);
	float out6 = rsGetElementAt_float(bias, m + 6 * parallelOFMs);
	float out7 = rsGetElementAt_float(bias, m + 7 * parallelOFMs);
	float out8 = rsGetElementAt_float(bias, m + 8 * parallelOFMs);
	float out9 = rsGetElementAt_float(bias, m + 9 * parallelOFMs);
	float out10 = rsGetElementAt_float(bias, m + 10 * parallelOFMs);
	float out11 = rsGetElementAt_float(bias, m + 11 * parallelOFMs);
	float out12 = rsGetElementAt_float(bias, m + 12 * parallelOFMs);
	float out13 = rsGetElementAt_float(bias, m + 13 * parallelOFMs);
	float out14 = rsGetElementAt_float(bias, m + 14 * parallelOFMs);
	float out15 = rsGetElementAt_float(bias, m + 15 * parallelOFMs);
	float out16 = rsGetElementAt_float(bias, m + 16 * parallelOFMs);
	float out17 = rsGetElementAt_float(bias, m + 17 * parallelOFMs);
	float out18 = rsGetElementAt_float(bias, m + 18 * parallelOFMs);
	float out19 = rsGetElementAt_float(bias, m + 19 * parallelOFMs);
	float out20 = rsGetElementAt_float(bias, m + 20 * parallelOFMs);
	float out21 = rsGetElementAt_float(bias, m + 21 * parallelOFMs);
	float out22 = rsGetElementAt_float(bias, m + 22 * parallelOFMs);
	float out23 = rsGetElementAt_float(bias, m + 23 * parallelOFMs);
	float out24 = rsGetElementAt_float(bias, m + 24 * parallelOFMs);
	float out25 = rsGetElementAt_float(bias, m + 25 * parallelOFMs);
	float out26 = rsGetElementAt_float(bias, m + 26 * parallelOFMs);
	float out27 = rsGetElementAt_float(bias, m + 27 * parallelOFMs);
	float out28 = rsGetElementAt_float(bias, m + 28 * parallelOFMs);
	float out29 = rsGetElementAt_float(bias, m + 29 * parallelOFMs);
	float out30 = rsGetElementAt_float(bias, m + 30 * parallelOFMs);
	float out31 = rsGetElementAt_float(bias, m + 31 * parallelOFMs);
	float out32 = rsGetElementAt_float(bias, m + 32 * parallelOFMs);
	float out33 = rsGetElementAt_float(bias, m + 33 * parallelOFMs);
	float out34 = rsGetElementAt_float(bias, m + 34 * parallelOFMs);
	float out35 = rsGetElementAt_float(bias, m + 35 * parallelOFMs);
	float out36 = rsGetElementAt_float(bias, m + 36 * parallelOFMs);
	float out37 = rsGetElementAt_float(bias, m + 37 * parallelOFMs);
	float out38 = rsGetElementAt_float(bias, m + 38 * parallelOFMs);
	float out39 = rsGetElementAt_float(bias, m + 39 * parallelOFMs);
	float out40 = rsGetElementAt_float(bias, m + 40 * parallelOFMs);
	float out41 = rsGetElementAt_float(bias, m + 41 * parallelOFMs);
	float out42 = rsGetElementAt_float(bias, m + 42 * parallelOFMs);
	float out43 = rsGetElementAt_float(bias, m + 43 * parallelOFMs);
	float out44 = rsGetElementAt_float(bias, m + 44 * parallelOFMs);
	float out45 = rsGetElementAt_float(bias, m + 45 * parallelOFMs);
	float out46 = rsGetElementAt_float(bias, m + 46 * parallelOFMs);
	float out47 = rsGetElementAt_float(bias, m + 47 * parallelOFMs);
	float out48 = rsGetElementAt_float(bias, m + 48 * parallelOFMs);
	float out49 = rsGetElementAt_float(bias, m + 49 * parallelOFMs);
	float out50 = rsGetElementAt_float(bias, m + 50 * parallelOFMs);
	float out51 = rsGetElementAt_float(bias, m + 51 * parallelOFMs);
	float out52 = rsGetElementAt_float(bias, m + 52 * parallelOFMs);
	float out53 = rsGetElementAt_float(bias, m + 53 * parallelOFMs);
	float out54 = rsGetElementAt_float(bias, m + 54 * parallelOFMs);
	float out55 = rsGetElementAt_float(bias, m + 55 * parallelOFMs);
	float out56 = rsGetElementAt_float(bias, m + 56 * parallelOFMs);
	float out57 = rsGetElementAt_float(bias, m + 57 * parallelOFMs);
	float out58 = rsGetElementAt_float(bias, m + 58 * parallelOFMs);
	float out59 = rsGetElementAt_float(bias, m + 59 * parallelOFMs);
	float out60 = rsGetElementAt_float(bias, m + 60 * parallelOFMs);
	float out61 = rsGetElementAt_float(bias, m + 61 * parallelOFMs);
	float out62 = rsGetElementAt_float(bias, m + 62 * parallelOFMs);
	float out63 = rsGetElementAt_float(bias, m + 63 * parallelOFMs);
	float out64 = rsGetElementAt_float(bias, m + 64 * parallelOFMs);
	float out65 = rsGetElementAt_float(bias, m + 65 * parallelOFMs);
	float out66 = rsGetElementAt_float(bias, m + 66 * parallelOFMs);
	float out67 = rsGetElementAt_float(bias, m + 67 * parallelOFMs);
	float out68 = rsGetElementAt_float(bias, m + 68 * parallelOFMs);
	float out69 = rsGetElementAt_float(bias, m + 69 * parallelOFMs);
	float out70 = rsGetElementAt_float(bias, m + 70 * parallelOFMs);
	float out71 = rsGetElementAt_float(bias, m + 71 * parallelOFMs);
	float out72 = rsGetElementAt_float(bias, m + 72 * parallelOFMs);
	float out73 = rsGetElementAt_float(bias, m + 73 * parallelOFMs);
	float out74 = rsGetElementAt_float(bias, m + 74 * parallelOFMs);
	float out75 = rsGetElementAt_float(bias, m + 75 * parallelOFMs);
	float out76 = rsGetElementAt_float(bias, m + 76 * parallelOFMs);
	float out77 = rsGetElementAt_float(bias, m + 77 * parallelOFMs);
	float out78 = rsGetElementAt_float(bias, m + 78 * parallelOFMs);
	float out79 = rsGetElementAt_float(bias, m + 79 * parallelOFMs);
	float out80 = rsGetElementAt_float(bias, m + 80 * parallelOFMs);
	float out81 = rsGetElementAt_float(bias, m + 81 * parallelOFMs);
	float out82 = rsGetElementAt_float(bias, m + 82 * parallelOFMs);
	float out83 = rsGetElementAt_float(bias, m + 83 * parallelOFMs);
	float out84 = rsGetElementAt_float(bias, m + 84 * parallelOFMs);
	float out85 = rsGetElementAt_float(bias, m + 85 * parallelOFMs);
	float out86 = rsGetElementAt_float(bias, m + 86 * parallelOFMs);
	float out87 = rsGetElementAt_float(bias, m + 87 * parallelOFMs);
	float out88 = rsGetElementAt_float(bias, m + 88 * parallelOFMs);
	float out89 = rsGetElementAt_float(bias, m + 89 * parallelOFMs);
	float out90 = rsGetElementAt_float(bias, m + 90 * parallelOFMs);
	float out91 = rsGetElementAt_float(bias, m + 91 * parallelOFMs);
	float out92 = rsGetElementAt_float(bias, m + 92 * parallelOFMs);
	float out93 = rsGetElementAt_float(bias, m + 93 * parallelOFMs);
	float out94 = rsGetElementAt_float(bias, m + 94 * parallelOFMs);
	float out95 = rsGetElementAt_float(bias, m + 95 * parallelOFMs);

	int32_t base1 = K * K * N_new;
	for (n = 0; n < N_new; n++){
		for (i = 0; i < K; i++){
			for (j = 0; j < K; j++){
				if (w*S + i - pad<0 || w*S + i - pad >= Win ||h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
				int32_t base2 = (i) + (K * j) + (K * K * n);

				float4 ifm = rsGetElementAt_float4(in, (w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n));

				float4 wght0 = rsGetElementAt_float4(weight, base2 + (base1 * m));
				out0 += dot(ifm, wght0);

				float4 wght1 = rsGetElementAt_float4(weight, base2 + (base1 * (m + parallelOFMs)));
				out1 += dot(ifm, wght1);

				float4 wght2 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 2 * parallelOFMs)));
				out2 += dot(ifm, wght2);

				float4 wght3 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 3 * parallelOFMs)));
				out3 += dot(ifm, wght3);

				float4 wght4 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 4 * parallelOFMs)));
				out4 += dot(ifm, wght4);

				float4 wght5 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 5 * parallelOFMs)));
				out5 += dot(ifm, wght5);

				float4 wght6 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 6 * parallelOFMs)));
				out6 += dot(ifm, wght6);

				float4 wght7 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 7 * parallelOFMs)));
				out7 += dot(ifm, wght7);

				float4 wght8 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 8 * parallelOFMs)));
				out8 += dot(ifm, wght8);

				float4 wght9 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 9 * parallelOFMs)));
				out9 += dot(ifm, wght9);

				float4 wght10 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 10 * parallelOFMs)));
				out10 += dot(ifm, wght10);

				float4 wght11 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 11 * parallelOFMs)));
				out11 += dot(ifm, wght11);

				float4 wght12 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 12 * parallelOFMs)));
				out12 += dot(ifm, wght12);

				float4 wght13 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 13 * parallelOFMs)));
				out13 += dot(ifm, wght13);

				float4 wght14 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 14 * parallelOFMs)));
				out14 += dot(ifm, wght14);

				float4 wght15 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 15 * parallelOFMs)));
				out15 += dot(ifm, wght15);

				float4 wght16 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 16 * parallelOFMs)));
				out16 += dot(ifm, wght16);

				float4 wght17 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 17 * parallelOFMs)));
				out17 += dot(ifm, wght17);

				float4 wght18 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 18 * parallelOFMs)));
				out18 += dot(ifm, wght18);

				float4 wght19 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 19 * parallelOFMs)));
				out19 += dot(ifm, wght19);

				float4 wght20 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 20 * parallelOFMs)));
				out20 += dot(ifm, wght20);

				float4 wght21 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 21 * parallelOFMs)));
				out21 += dot(ifm, wght21);

				float4 wght22 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 22 * parallelOFMs)));
				out22 += dot(ifm, wght22);

				float4 wght23 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 23 * parallelOFMs)));
				out23 += dot(ifm, wght23);

				float4 wght24 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 24 * parallelOFMs)));
				out24 += dot(ifm, wght24);

				float4 wght25 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 25 * parallelOFMs)));
				out25 += dot(ifm, wght25);

				float4 wght26 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 26 * parallelOFMs)));
				out26 += dot(ifm, wght26);

				float4 wght27 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 27 * parallelOFMs)));
				out27 += dot(ifm, wght27);

				float4 wght28 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 28 * parallelOFMs)));
				out28 += dot(ifm, wght28);

				float4 wght29 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 29 * parallelOFMs)));
				out29 += dot(ifm, wght29);

				float4 wght30 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 30 * parallelOFMs)));
				out30 += dot(ifm, wght30);

				float4 wght31 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 31 * parallelOFMs)));
				out31 += dot(ifm, wght31);

				float4 wght32 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 32 * parallelOFMs)));
				out32 += dot(ifm, wght32);

				float4 wght33 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 33 * parallelOFMs)));
				out33 += dot(ifm, wght33);

				float4 wght34 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 34 * parallelOFMs)));
				out34 += dot(ifm, wght34);

				float4 wght35 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 35 * parallelOFMs)));
				out35 += dot(ifm, wght35);

				float4 wght36 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 36 * parallelOFMs)));
				out36 += dot(ifm, wght36);

				float4 wght37 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 37 * parallelOFMs)));
				out37 += dot(ifm, wght37);

				float4 wght38 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 38 * parallelOFMs)));
				out38 += dot(ifm, wght38);

				float4 wght39 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 39 * parallelOFMs)));
				out39 += dot(ifm, wght39);

				float4 wght40 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 40 * parallelOFMs)));
				out40 += dot(ifm, wght40);

				float4 wght41 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 41 * parallelOFMs)));
				out41 += dot(ifm, wght41);

				float4 wght42 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 42 * parallelOFMs)));
				out42 += dot(ifm, wght42);

				float4 wght43 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 43 * parallelOFMs)));
				out43 += dot(ifm, wght43);

				float4 wght44 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 44 * parallelOFMs)));
				out44 += dot(ifm, wght44);

				float4 wght45 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 45 * parallelOFMs)));
				out45 += dot(ifm, wght45);

				float4 wght46 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 46 * parallelOFMs)));
				out46 += dot(ifm, wght46);

				float4 wght47 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 47 * parallelOFMs)));
				out47 += dot(ifm, wght47);

				float4 wght48 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 48 * parallelOFMs)));
				out48 += dot(ifm, wght48);

				float4 wght49 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 49 * parallelOFMs)));
				out49 += dot(ifm, wght49);

				float4 wght50 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 50 * parallelOFMs)));
				out50 += dot(ifm, wght50);

				float4 wght51 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 51 * parallelOFMs)));
				out51 += dot(ifm, wght51);

				float4 wght52 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 52 * parallelOFMs)));
				out52 += dot(ifm, wght52);

				float4 wght53 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 53 * parallelOFMs)));
				out53 += dot(ifm, wght53);

				float4 wght54 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 54 * parallelOFMs)));
				out54 += dot(ifm, wght54);

				float4 wght55 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 55 * parallelOFMs)));
				out55 += dot(ifm, wght55);

				float4 wght56 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 56 * parallelOFMs)));
				out56 += dot(ifm, wght56);

				float4 wght57 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 57 * parallelOFMs)));
				out57 += dot(ifm, wght57);

				float4 wght58 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 58 * parallelOFMs)));
				out58 += dot(ifm, wght58);

				float4 wght59 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 59 * parallelOFMs)));
				out59 += dot(ifm, wght59);

				float4 wght60 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 60 * parallelOFMs)));
				out60 += dot(ifm, wght60);

				float4 wght61 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 61 * parallelOFMs)));
				out61 += dot(ifm, wght61);

				float4 wght62 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 62 * parallelOFMs)));
				out62 += dot(ifm, wght62);

				float4 wght63 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 63 * parallelOFMs)));
				out63 += dot(ifm, wght63);

				float4 wght64 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 64 * parallelOFMs)));
				out64 += dot(ifm, wght64);

				float4 wght65 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 65 * parallelOFMs)));
				out65 += dot(ifm, wght65);

				float4 wght66 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 66 * parallelOFMs)));
				out66 += dot(ifm, wght66);

				float4 wght67 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 67 * parallelOFMs)));
				out67 += dot(ifm, wght67);

				float4 wght68 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 68 * parallelOFMs)));
				out68 += dot(ifm, wght68);

				float4 wght69 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 69 * parallelOFMs)));
				out69 += dot(ifm, wght69);

				float4 wght70 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 70 * parallelOFMs)));
				out70 += dot(ifm, wght70);

				float4 wght71 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 71 * parallelOFMs)));
				out71 += dot(ifm, wght71);

				float4 wght72 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 72 * parallelOFMs)));
				out72 += dot(ifm, wght72);

				float4 wght73 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 73 * parallelOFMs)));
				out73 += dot(ifm, wght73);

				float4 wght74 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 74 * parallelOFMs)));
				out74 += dot(ifm, wght74);

				float4 wght75 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 75 * parallelOFMs)));
				out75 += dot(ifm, wght75);

				float4 wght76 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 76 * parallelOFMs)));
				out76 += dot(ifm, wght76);

				float4 wght77 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 77 * parallelOFMs)));
				out77 += dot(ifm, wght77);

				float4 wght78 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 78 * parallelOFMs)));
				out78 += dot(ifm, wght78);

				float4 wght79 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 79 * parallelOFMs)));
				out79 += dot(ifm, wght79);

				float4 wght80 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 80 * parallelOFMs)));
				out80 += dot(ifm, wght80);

				float4 wght81 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 81 * parallelOFMs)));
				out81 += dot(ifm, wght81);

				float4 wght82 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 82 * parallelOFMs)));
				out82 += dot(ifm, wght82);

				float4 wght83 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 83 * parallelOFMs)));
				out83 += dot(ifm, wght83);

				float4 wght84 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 84 * parallelOFMs)));
				out84 += dot(ifm, wght84);

				float4 wght85 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 85 * parallelOFMs)));
				out85 += dot(ifm, wght85);

				float4 wght86 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 86 * parallelOFMs)));
				out86 += dot(ifm, wght86);

				float4 wght87 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 87 * parallelOFMs)));
				out87 += dot(ifm, wght87);

				float4 wght88 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 88 * parallelOFMs)));
				out88 += dot(ifm, wght88);

				float4 wght89 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 89 * parallelOFMs)));
				out89 += dot(ifm, wght89);

				float4 wght90 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 90 * parallelOFMs)));
				out90 += dot(ifm, wght90);

				float4 wght91 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 91 * parallelOFMs)));
				out91 += dot(ifm, wght91);

				float4 wght92 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 92 * parallelOFMs)));
				out92 += dot(ifm, wght92);

				float4 wght93 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 93 * parallelOFMs)));
				out93 += dot(ifm, wght93);

				float4 wght94 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 94 * parallelOFMs)));
				out94 += dot(ifm, wght94);

				float4 wght95 = rsGetElementAt_float4(weight, base2 + (base1 * (m + 95 * parallelOFMs)));
				out95 += dot(ifm, wght95);

			}
		}
	}

	int32_t base3 = offset + x;
	int32_t base4 = Wout * Hout * parallelOFMs;

	out0 = fmax(0.0f, out0);
	rsSetElementAt_float(output, out0, base3);

	out1 = fmax(0.0f, out1);
	rsSetElementAt_float(output, out1, base3 + base4);

	out2 = fmax(0.0f, out2);
	rsSetElementAt_float(output, out2, base3 + 2 * base4);

	out3 = fmax(0.0f, out3);
	rsSetElementAt_float(output, out3, base3 + 3 * base4);

	out4 = fmax(0.0f, out4);
	rsSetElementAt_float(output, out4, base3 + 4 * base4);

	out5 = fmax(0.0f, out5);
	rsSetElementAt_float(output, out5, base3 + 5 * base4);

	out6 = fmax(0.0f, out6);
	rsSetElementAt_float(output, out6, base3 + 6 * base4);

	out7 = fmax(0.0f, out7);
	rsSetElementAt_float(output, out7, base3 + 7 * base4);

	out8 = fmax(0.0f, out8);
	rsSetElementAt_float(output, out8, base3 + 8 * base4);

	out9 = fmax(0.0f, out9);
	rsSetElementAt_float(output, out9, base3 + 9 * base4);

	out10 = fmax(0.0f, out10);
	rsSetElementAt_float(output, out10, base3 + 10 * base4);

	out11 = fmax(0.0f, out11);
	rsSetElementAt_float(output, out11, base3 + 11 * base4);

	out12 = fmax(0.0f, out12);
	rsSetElementAt_float(output, out12, base3 + 12 * base4);

	out13 = fmax(0.0f, out13);
	rsSetElementAt_float(output, out13, base3 + 13 * base4);

	out14 = fmax(0.0f, out14);
	rsSetElementAt_float(output, out14, base3 + 14 * base4);

	out15 = fmax(0.0f, out15);
	rsSetElementAt_float(output, out15, base3 + 15 * base4);

	out16 = fmax(0.0f, out16);
	rsSetElementAt_float(output, out16, base3 + 16 * base4);

	out17 = fmax(0.0f, out17);
	rsSetElementAt_float(output, out17, base3 + 17 * base4);

	out18 = fmax(0.0f, out18);
	rsSetElementAt_float(output, out18, base3 + 18 * base4);

	out19 = fmax(0.0f, out19);
	rsSetElementAt_float(output, out19, base3 + 19 * base4);

	out20 = fmax(0.0f, out20);
	rsSetElementAt_float(output, out20, base3 + 20 * base4);

	out21 = fmax(0.0f, out21);
	rsSetElementAt_float(output, out21, base3 + 21 * base4);

	out22 = fmax(0.0f, out22);
	rsSetElementAt_float(output, out22, base3 + 22 * base4);

	out23 = fmax(0.0f, out23);
	rsSetElementAt_float(output, out23, base3 + 23 * base4);

	out24 = fmax(0.0f, out24);
	rsSetElementAt_float(output, out24, base3 + 24 * base4);

	out25 = fmax(0.0f, out25);
	rsSetElementAt_float(output, out25, base3 + 25 * base4);

	out26 = fmax(0.0f, out26);
	rsSetElementAt_float(output, out26, base3 + 26 * base4);

	out27 = fmax(0.0f, out27);
	rsSetElementAt_float(output, out27, base3 + 27 * base4);

	out28 = fmax(0.0f, out28);
	rsSetElementAt_float(output, out28, base3 + 28 * base4);

	out29 = fmax(0.0f, out29);
	rsSetElementAt_float(output, out29, base3 + 29 * base4);

	out30 = fmax(0.0f, out30);
	rsSetElementAt_float(output, out30, base3 + 30 * base4);

	out31 = fmax(0.0f, out31);
	rsSetElementAt_float(output, out31, base3 + 31 * base4);

	out32 = fmax(0.0f, out32);
	rsSetElementAt_float(output, out32, base3 + 32 * base4);

	out33 = fmax(0.0f, out33);
	rsSetElementAt_float(output, out33, base3 + 33 * base4);

	out34 = fmax(0.0f, out34);
	rsSetElementAt_float(output, out34, base3 + 34 * base4);

	out35 = fmax(0.0f, out35);
	rsSetElementAt_float(output, out35, base3 + 35 * base4);

	out36 = fmax(0.0f, out36);
	rsSetElementAt_float(output, out36, base3 + 36 * base4);

	out37 = fmax(0.0f, out37);
	rsSetElementAt_float(output, out37, base3 + 37 * base4);

	out38 = fmax(0.0f, out38);
	rsSetElementAt_float(output, out38, base3 + 38 * base4);

	out39 = fmax(0.0f, out39);
	rsSetElementAt_float(output, out39, base3 + 39 * base4);

	out40 = fmax(0.0f, out40);
	rsSetElementAt_float(output, out40, base3 + 40 * base4);

	out41 = fmax(0.0f, out41);
	rsSetElementAt_float(output, out41, base3 + 41 * base4);

	out42 = fmax(0.0f, out42);
	rsSetElementAt_float(output, out42, base3 + 42 * base4);

	out43 = fmax(0.0f, out43);
	rsSetElementAt_float(output, out43, base3 + 43 * base4);

	out44 = fmax(0.0f, out44);
	rsSetElementAt_float(output, out44, base3 + 44 * base4);

	out45 = fmax(0.0f, out45);
	rsSetElementAt_float(output, out45, base3 + 45 * base4);

	out46 = fmax(0.0f, out46);
	rsSetElementAt_float(output, out46, base3 + 46 * base4);

	out47 = fmax(0.0f, out47);
	rsSetElementAt_float(output, out47, base3 + 47 * base4);

	out48 = fmax(0.0f, out48);
	rsSetElementAt_float(output, out48, base3 + 48 * base4);

	out49 = fmax(0.0f, out49);
	rsSetElementAt_float(output, out49, base3 + 49 * base4);

	out50 = fmax(0.0f, out50);
	rsSetElementAt_float(output, out50, base3 + 50 * base4);

	out51 = fmax(0.0f, out51);
	rsSetElementAt_float(output, out51, base3 + 51 * base4);

	out52 = fmax(0.0f, out52);
	rsSetElementAt_float(output, out52, base3 + 52 * base4);

	out53 = fmax(0.0f, out53);
	rsSetElementAt_float(output, out53, base3 + 53 * base4);

	out54 = fmax(0.0f, out54);
	rsSetElementAt_float(output, out54, base3 + 54 * base4);

	out55 = fmax(0.0f, out55);
	rsSetElementAt_float(output, out55, base3 + 55 * base4);

	out56 = fmax(0.0f, out56);
	rsSetElementAt_float(output, out56, base3 + 56 * base4);

	out57 = fmax(0.0f, out57);
	rsSetElementAt_float(output, out57, base3 + 57 * base4);

	out58 = fmax(0.0f, out58);
	rsSetElementAt_float(output, out58, base3 + 58 * base4);

	out59 = fmax(0.0f, out59);
	rsSetElementAt_float(output, out59, base3 + 59 * base4);

	out60 = fmax(0.0f, out60);
	rsSetElementAt_float(output, out60, base3 + 60 * base4);

	out61 = fmax(0.0f, out61);
	rsSetElementAt_float(output, out61, base3 + 61 * base4);

	out62 = fmax(0.0f, out62);
	rsSetElementAt_float(output, out62, base3 + 62 * base4);

	out63 = fmax(0.0f, out63);
	rsSetElementAt_float(output, out63, base3 + 63 * base4);

	out64 = fmax(0.0f, out64);
	rsSetElementAt_float(output, out64, base3 + 64 * base4);

	out65 = fmax(0.0f, out65);
	rsSetElementAt_float(output, out65, base3 + 65 * base4);

	out66 = fmax(0.0f, out66);
	rsSetElementAt_float(output, out66, base3 + 66 * base4);

	out67 = fmax(0.0f, out67);
	rsSetElementAt_float(output, out67, base3 + 67 * base4);

	out68 = fmax(0.0f, out68);
	rsSetElementAt_float(output, out68, base3 + 68 * base4);

	out69 = fmax(0.0f, out69);
	rsSetElementAt_float(output, out69, base3 + 69 * base4);

	out70 = fmax(0.0f, out70);
	rsSetElementAt_float(output, out70, base3 + 70 * base4);

	out71 = fmax(0.0f, out71);
	rsSetElementAt_float(output, out71, base3 + 71 * base4);

	out72 = fmax(0.0f, out72);
	rsSetElementAt_float(output, out72, base3 + 72 * base4);

	out73 = fmax(0.0f, out73);
	rsSetElementAt_float(output, out73, base3 + 73 * base4);

	out74 = fmax(0.0f, out74);
	rsSetElementAt_float(output, out74, base3 + 74 * base4);

	out75 = fmax(0.0f, out75);
	rsSetElementAt_float(output, out75, base3 + 75 * base4);

	out76 = fmax(0.0f, out76);
	rsSetElementAt_float(output, out76, base3 + 76 * base4);

	out77 = fmax(0.0f, out77);
	rsSetElementAt_float(output, out77, base3 + 77 * base4);

	out78 = fmax(0.0f, out78);
	rsSetElementAt_float(output, out78, base3 + 78 * base4);

	out79 = fmax(0.0f, out79);
	rsSetElementAt_float(output, out79, base3 + 79 * base4);

	out80 = fmax(0.0f, out80);
	rsSetElementAt_float(output, out80, base3 + 80 * base4);

	out81 = fmax(0.0f, out81);
	rsSetElementAt_float(output, out81, base3 + 81 * base4);

	out82 = fmax(0.0f, out82);
	rsSetElementAt_float(output, out82, base3 + 82 * base4);

	out83 = fmax(0.0f, out83);
	rsSetElementAt_float(output, out83, base3 + 83 * base4);

	out84 = fmax(0.0f, out84);
	rsSetElementAt_float(output, out84, base3 + 84 * base4);

	out85 = fmax(0.0f, out85);
	rsSetElementAt_float(output, out85, base3 + 85 * base4);

	out86 = fmax(0.0f, out86);
	rsSetElementAt_float(output, out86, base3 + 86 * base4);

	out87 = fmax(0.0f, out87);
	rsSetElementAt_float(output, out87, base3 + 87 * base4);

	out88 = fmax(0.0f, out88);
	rsSetElementAt_float(output, out88, base3 + 88 * base4);

	out89 = fmax(0.0f, out89);
	rsSetElementAt_float(output, out89, base3 + 89 * base4);

	out90 = fmax(0.0f, out90);
	rsSetElementAt_float(output, out90, base3 + 90 * base4);

	out91 = fmax(0.0f, out91);
	rsSetElementAt_float(output, out91, base3 + 91 * base4);

	out92 = fmax(0.0f, out92);
	rsSetElementAt_float(output, out92, base3 + 92 * base4);

	out93 = fmax(0.0f, out93);
	rsSetElementAt_float(output, out93, base3 + 93 * base4);

	out94 = fmax(0.0f, out94);
	rsSetElementAt_float(output, out94, base3 + 94 * base4);

	out95 = fmax(0.0f, out95);
	rsSetElementAt_float(output, out95, base3 + 95 * base4);

}
