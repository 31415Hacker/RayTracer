#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <SDL2/SDL.h>
#include <cblas.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#ifndef M_PI
#define M_PI 3.1415926535
#endif

// ──────── HYPERPARAMS & AUGMENTATION ────────

// dataset image dims
#define IMG_W 28
#define IMG_H 28

// defaults—override with --batch= --lr= --epochs=
static int BATCH_SIZE = 512;
static float LEARNING_RATE = 0.01f;
static int EPOCHS = 50;

// data file & save path
#define DATAFILE "mnist_train.dat"
#define SAVEFILE "network.dat"

// augmentation noise + scale + shift
#define AUG_NOISE_STD 0.1f
#define AUG_SCALE_MIN 0.9f
#define AUG_SCALE_MAX 1.1f
#define AUG_SHIFT_MAX 5

// SDL window
#define WIN_W 640
#define WIN_H 480

// ──────── UTILITIES ────────
static jmp_buf jump_env;

void handle_sigint(int sig)
{
    fprintf(stderr, "\n[!] Caught signal %d — exiting immediately.\n", sig);
    longjmp(jump_env, 1);
}

// aligned allocator
static void *aligned_malloc(size_t bytes, size_t align)
{
    void *p;
    return posix_memalign(&p, align, bytes) == 0 ? p : NULL;
}

// simple Box–Muller → N(0,1)
static float rand_normal()
{
    float u = (rand() + 1.f) / (RAND_MAX + 2.f);
    float v = rand() / (float)RAND_MAX;
    return sqrtf(-2.f * logf(u)) * cosf(2.f * M_PI * v);
}

// ──────── SDL2 GRAPHING ────────

static SDL_Window *window = NULL;
static SDL_Renderer *renderer = NULL;

static void init_graph()
{
    SDL_Init(SDL_INIT_VIDEO);
    window = SDL_CreateWindow("Training Monitor",
                              SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                              WIN_W, WIN_H, 0);
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
}

static void update_graph(
    float *train_loss, float *train_acc,
    float *test_loss, float *test_acc,
    int epoch)
{
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderDrawLine(renderer, 40, WIN_H - 40, WIN_W - 10, WIN_H - 40);
    SDL_RenderDrawLine(renderer, 40, 10, 40, WIN_H - 40);

    float xs = (WIN_W - 50) / (float)epoch;
    float ys = (WIN_H - 50) / (float)train_loss[0];

    for (int i = 1; i < epoch; i++)
    {
        int x0 = 40 + (int)(xs * (i - 1)), x1 = 40 + (int)(xs * i);
        // loss = red
        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
        int y0l = WIN_H - 40 - (int)(train_loss[i - 1] * ys),
            y1l = WIN_H - 40 - (int)(train_loss[i] * ys);
        SDL_RenderDrawLine(renderer, x0, y0l, x1, y1l);
        // train acc = green
        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
        int y0a = WIN_H - 40 - (int)(train_acc[i - 1] * ys),
            y1a = WIN_H - 40 - (int)(train_acc[i] * ys);
        SDL_RenderDrawLine(renderer, x0, y0a, x1, y1a);
        // test acc = blue
        SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255);
        int y0t = WIN_H - 40 - (int)(test_acc[i - 1] * ys),
            y1t = WIN_H - 40 - (int)(test_acc[i] * ys);
        SDL_RenderDrawLine(renderer, x0, y0t, x1, y1t);
    }
    SDL_RenderPresent(renderer);
}

static void shutdown_graph()
{
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

// ──────── AUGMENTATION ────────

static void augment_image(const float *src, float *dst)
{
    float scale = AUG_SCALE_MIN + (rand() / (float)RAND_MAX) * (AUG_SCALE_MAX - AUG_SCALE_MIN);
    int sx = (rand() % (2 * AUG_SHIFT_MAX + 1)) - AUG_SHIFT_MAX;
    int sy = (rand() % (2 * AUG_SHIFT_MAX + 1)) - AUG_SHIFT_MAX;
    const float cx = (IMG_W - 1) / 2.f, cy = (IMG_H - 1) / 2.f;
    for (int y = 0; y < IMG_H; y++)
        for (int x = 0; x < IMG_W; x++)
        {
            float u = (x - cx) / scale + cx - sx;
            float v = (y - cy) / scale + cy - sy;
            int iu = (int)roundf(u), iv = (int)roundf(v);
            float val = 0;
            if (iu >= 0 && iu < IMG_W && iv >= 0 && iv < IMG_H)
                val = src[iv * IMG_W + iu];
            val += rand_normal() * AUG_NOISE_STD;
            dst[y * IMG_W + x] = val < 0 ? 0 : (val > 1 ? 1 : val);
        }
}

static void augment_batch(const float *X, float *X_aug, int bs)
{
    for (int i = 0; i < bs; i++)
        augment_image(X + i * IMG_W * IMG_H,
                      X_aug + i * IMG_W * IMG_H);
}

// ──────── DATA LOADING ────────

static int load_dataset(const char *path, float **Xptr, float **Yptr)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        perror("open");
        exit(1);
    }
    int N;
    fread(&N, sizeof(int), 1, f);
    float *X = aligned_malloc((size_t)N * IMG_W * IMG_H * sizeof(float), 64);
    float *Y = aligned_malloc((size_t)N * 10 * sizeof(float), 64);
    for (int i = 0; i < N; i++)
    {
        fread(X + (size_t)i * IMG_W * IMG_H, sizeof(float), IMG_W * IMG_H, f);
        for (int j = 0; j < IMG_W * IMG_H; j++)
            X[i * IMG_W * IMG_H + j] /= 255.f;
        float lbl;
        fread(&lbl, sizeof(float), 1, f);
        for (int j = 0; j < 10; j++)
            Y[i * 10 + j] = (j == (int)lbl) ? 1.f : 0.f;
    }
    fclose(f);
    *Xptr = X;
    *Yptr = Y;
    return N;
}

static void shuffle(int N, float *X, float *Y)
{
    float *tmpX = aligned_malloc(IMG_W * IMG_H * sizeof(float), 64),
          *tmpY = aligned_malloc(10 * sizeof(float), 64);
    for (int i = N - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        memcpy(tmpX, X + i * IMG_W * IMG_H, IMG_W * IMG_H * sizeof(float));
        memcpy(X + i * IMG_W * IMG_H, X + j * IMG_W * IMG_H, IMG_W * IMG_H * sizeof(float));
        memcpy(X + j * IMG_W * IMG_H, tmpX, IMG_W * IMG_H * sizeof(float));
        memcpy(tmpY, Y + i * 10, 10 * sizeof(float));
        memcpy(Y + i * 10, Y + j * 10, 10 * sizeof(float));
        memcpy(Y + j * 10, tmpY, 10 * sizeof(float));
    }
    free(tmpX);
    free(tmpY);
}

// ──────── NETWORK STRUCT & I/O ────────

typedef struct
{
    int L;          // # layers
    int *sz;        // sizes
    float **W, **b; // W[1..L-1], b[1..L-1]
} Net;

static void net_init(Net *n, int L, int *sz)
{
    n->L = L;
    n->sz = malloc(L * sizeof(int));
    memcpy(n->sz, sz, L * sizeof(int));
    n->W = malloc(L * sizeof(float *));
    n->b = malloc(L * sizeof(float *));
    for (int l = 1; l < L; l++)
    {
        size_t S = (size_t)sz[l] * sz[l - 1];
        n->W[l] = aligned_malloc(S * sizeof(float), 64);
        n->b[l] = aligned_malloc(sz[l] * sizeof(float), 64);
        for (size_t i = 0; i < S; i++)
            n->W[l][i] = (rand() / (float)RAND_MAX) * 2 - 1;
        for (int i = 0; i < sz[l]; i++)
            n->b[l][i] = (rand() / (float)RAND_MAX) * 2 - 1;
    }
}

static int net_load(Net *n, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return 0;
    int L;
    fread(&L, sizeof(int), 1, f);
    int *sz = malloc(L * sizeof(int));
    fread(sz, sizeof(int), L, f);
    net_init(n, L, sz);
    for (int l = 1; l < L; l++)
    {
        size_t S = (size_t)sz[l] * sz[l - 1];
        fread(n->W[l], sizeof(float), S, f);
        fread(n->b[l], sizeof(float), sz[l], f);
    }
    free(sz);
    fclose(f);
    return 1;
}

static void net_save(Net *n, const char *path)
{
    FILE *f = fopen(path, "wb");
    fwrite(&n->L, sizeof(int), 1, f);
    fwrite(n->sz, sizeof(int), n->L, f);
    for (int l = 1; l < n->L; l++)
    {
        size_t S = (size_t)n->sz[l] * n->sz[l - 1];
        fwrite(n->W[l], sizeof(float), S, f);
        fwrite(n->b[l], sizeof(float), n->sz[l], f);
    }
    fclose(f);
}

static void net_free(Net *n)
{
    for (int l = 1; l < n->L; l++)
    {
        free(n->W[l]);
        free(n->b[l]);
    }
    free(n->W);
    free(n->b);
    free(n->sz);
}

// ──────── TRAIN & EVAL ────────

static float relu(float x)
{
    return x > 0 ? x : 0;
}
static float drelu(float x)
{
    return x > 0 ? 1 : 0;
}

static void softmax_and_metrics(
    float *A, const float *Y, int bs, int dim,
    double *loss, int *correct)
{
    for (int n = 0; n < bs; n++)
    {
        float mx = A[n * dim];
        for (int j = 1; j < dim; j++)
            if (A[n * dim + j] > mx)
                mx = A[n * dim + j];
        float sum = 0;
        for (int j = 0; j < dim; j++)
        {
            A[n * dim + j] = expf(A[n * dim + j] - mx);
            sum += A[n * dim + j];
        }
        int pred = 0, act = 0;
        for (int j = 0; j < dim; j++)
        {
            float p = A[n * dim + j] /= sum;
            if (p > A[n * dim + pred])
                pred = j;
            if (Y[n * dim + j] > 0.5f)
                act = j;
        }
        if (pred == act)
            (*correct)++;
        *loss += -logf(fmaxf(A[n * dim + act], 1e-7f));
    }
}

static void train_epoch(
    Net *net,
    float *Xb, float *Xb_aug,
    float *Yb, int Ntr,
    float **A, float **dZ, float **dW, float **db, float **dA,
    double *out_loss, double *out_acc,
    float LEARNING_RATE, int BATCH_SIZE)
{
    int L = net->L, *sz = net->sz;
    double total_loss = 0;
    int total_correct = 0;
    for (int i = 0; i < Ntr; i += BATCH_SIZE)
    {
        int bs = i + BATCH_SIZE > Ntr ? Ntr - i : BATCH_SIZE;
        augment_batch(Xb + i * IMG_W * IMG_H, Xb_aug, bs);
        A[0] = Xb_aug;
        for (int l = 1; l < L; l++)
        {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        bs, sz[l], sz[l - 1],
                        1.f, A[l - 1], sz[l - 1],
                        net->W[l], sz[l],
                        0.f, A[l], sz[l]);
            for (int n = 0; n < bs; n++)
                for (int j = 0; j < sz[l]; j++)
                {
                    float v = A[l][n * sz[l] + j] + net->b[l][j];
                    A[l][n * sz[l] + j] = (l == L - 1 ? v : relu(v));
                }
        }
        softmax_and_metrics(A[L - 1], Yb + i * sz[L - 1], bs, sz[L - 1],
                            &total_loss, &total_correct);
        // backprop
        for (int k = 0; k < bs * sz[L - 1]; k++)
            dZ[L - 1][k] = A[L - 1][k] - Yb[i * sz[L - 1] + k];
        for (int l = L - 1; l >= 1; l--)
        {
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        sz[l - 1], sz[l], bs,
                        1.f / bs, A[l - 1], sz[l - 1],
                        dZ[l], sz[l],
                        0.f, dW[l], sz[l]);
            for (int j = 0; j < sz[l]; j++)
            {
                float s = 0;
                for (int n = 0; n < bs; n++)
                    s += dZ[l][n * sz[l] + j];
                db[l][j] = s / bs;
            }
            if (l > 1)
            {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            bs, sz[l - 1], sz[l],
                            1.f, dZ[l], sz[l],
                            net->W[l], sz[l],
                            0.f, dA[l - 1], sz[l - 1]);
                for (int k = 0; k < bs * sz[l - 1]; k++)
                    dZ[l - 1][k] = dA[l - 1][k] * drelu(A[l - 1][k]);
            }
            cblas_saxpy((size_t)sz[l - 1] * sz[l], -LEARNING_RATE, dW[l], 1, net->W[l], 1);
            cblas_saxpy(sz[l], -LEARNING_RATE, db[l], 1, net->b[l], 1);
        }
    }
    *out_loss = total_loss;
    *out_acc = total_correct;
}

static void eval_epoch(
    Net *net,
    float *Xte, float *Yte, int Nte,
    float **A,
    double *out_loss, double *out_acc,
    int BATCH_SIZE)
{
    int L = net->L, *sz = net->sz;
    double loss = 0;
    int correct = 0;
    for (int i = 0; i < Nte; i += BATCH_SIZE)
    {
        int bs = i + BATCH_SIZE > Nte ? Nte - i : BATCH_SIZE;
        A[0] = Xte + i * IMG_W * IMG_H;
        for (int l = 1; l < L; l++)
        {
            int prevdim = (l == 1 ? IMG_W * IMG_H : sz[l - 1]);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        bs, sz[l], prevdim,
                        1.f, A[l - 1], prevdim,
                        net->W[l], sz[l],
                        0.f, A[l], sz[l]);
            for (int n = 0; n < bs; n++)
                for (int j = 0; j < sz[l]; j++)
                {
                    float v = A[l][n * sz[l] + j] + net->b[l][j];
                    A[l][n * sz[l] + j] = (l == L - 1 ? v : relu(v));
                }
        }
        softmax_and_metrics(A[L - 1], Yte + i * sz[L - 1], bs, sz[L - 1], &loss, &correct);
    }
    *out_loss = loss;
    *out_acc = correct;
}

// 1) set up Ctrl+C & kill handlers
static void setup_signal_handlers(void)
{
    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);
}

// 2) SDL event thread jumps out on window close
static void *sdl_event_thread(void *arg)
{
    SDL_Event e;
    while (SDL_WaitEvent(&e))
    {
        if (e.type == SDL_QUIT)
            longjmp(jump_env, 1);
    }
    return NULL;
}

// 3) Load & split 80/20
static void load_and_split_dataset(
    float **Xtr, float **Ytr, int *Ntr,
    float **Xte, float **Yte, int *Nte)
{
    float *X, *Y;
    int N = load_dataset(DATAFILE, &X, &Y);
    shuffle(N, X, Y);
    *Ntr = N * 8 / 10;
    *Nte = N - *Ntr;
    *Xtr = X;
    *Ytr = Y;
    *Xte = X + (*Ntr) * IMG_W * IMG_H;
    *Yte = Y + (*Ntr) * 10;
}

// 4) Allocate A, dZ, dW, db, dA arrays
static void allocate_buffers(int L, int *sizes,
                             float ***A, float ***dZ,
                             float ***dW, float ***db,
                             float ***dA)
{
    *A = malloc(L * sizeof(float *));
    *dZ = malloc(L * sizeof(float *));
    *dA = malloc(L * sizeof(float *));
    *dW = malloc(L * sizeof(float *));
    *db = malloc(L * sizeof(float *));
    for (int l = 0; l < L; l++)
    {
        int dim = (l == 0 ? IMG_W * IMG_H : sizes[l]);
        (*A)[l] = aligned_malloc(BATCH_SIZE * dim * sizeof(float), 64);
        (*dZ)[l] = aligned_malloc(BATCH_SIZE * dim * sizeof(float), 64);
        (*dA)[l] = aligned_malloc(BATCH_SIZE * dim * sizeof(float), 64);
        if (l > 0)
        {
            size_t S = (size_t)sizes[l] * sizes[l - 1];
            (*dW)[l] = aligned_malloc(S * sizeof(float), 64);
            (*db)[l] = aligned_malloc(sizes[l] * sizeof(float), 64);
        }
    }
}

// 5) Free those buffers
static void free_buffers(int L,
                         float **A, float **dZ,
                         float **dW, float **db,
                         float **dA)
{
    for (int l = 0; l < L; l++)
    {
        free(A[l]);
        free(dZ[l]);
        free(dA[l]);
        if (l > 0)
        {
            free(dW[l]);
            free(db[l]);
        }
    }
    free(A);
    free(dZ);
    free(dW);
    free(db);
    free(dA);
}

// 6) Load existing or init new network
static void load_or_init_net(Net *net, int run_test_only)
{
    int sizes[] = {IMG_W * IMG_H, 256, 128, 10};
    int L = sizeof(sizes) / sizeof(int);
    if (access(SAVEFILE, 0) == 0 && net_load(net, SAVEFILE))
    {
        printf("🗄 Loaded saved network\n");
    }
    else if (run_test_only)
    {
        fprintf(stderr, "❌ No saved model so can't test\n");
        exit(1);
    }
    else
    {
        net_init(net, L, sizes);
    }
}

// 7) The training loop
static void train_loop(Net *net,
                       float *Xtr, float *Ytr, int Ntr,
                       float *Xte, float *Yte, int Nte,
                       float **A, float **dZ,
                       float **dW, float **db,
                       float **dA)
{
    char buf[32];
    getchar(); // Catch the newline, preventing it from making the learning rate empty.
    printf("Enter Learning Rate: \n");
    if (!fgets(buf, sizeof buf, stdin))
    {
        fprintf(stderr, "Input error\n");
        return;
    }
    LEARNING_RATE = strtof(buf, NULL);

    printf("Enter Batch Size: \n");
    if (!fgets(buf, sizeof buf, stdin))
    {
        fprintf(stderr, "Input error\n");
        return;
    }
    BATCH_SIZE = strtof(buf, NULL);

    printf("Enter number of epochs to train: \n");
    if (!fgets(buf, sizeof buf, stdin))
    {
        fprintf(stderr, "Input error\n");
        return;
    }
    EPOCHS = strtof(buf, NULL);

    float *Xb_aug = aligned_malloc(BATCH_SIZE * IMG_W * IMG_H * sizeof(float), 64);
    float *t_loss = calloc(EPOCHS, sizeof(float));
    float *t_acc = calloc(EPOCHS, sizeof(float));
    float *v_loss = calloc(EPOCHS, sizeof(float));
    float *v_acc = calloc(EPOCHS, sizeof(float));

    for (int ep = 0; ep < EPOCHS; ep++)
    {
        double Ltr, Atr, Lte, Ate;
        train_epoch(net, Xtr, Xb_aug, Ytr, Ntr,
                    A, dZ, dW, db, dA, &Ltr, &Atr,
                    LEARNING_RATE, BATCH_SIZE);
        eval_epoch(net, Xte, Yte, Nte,
                   A, &Lte, &Ate, 512);

        t_loss[ep] = Ltr / Ntr;
        t_acc[ep] = Atr / Ntr;
        v_loss[ep] = Lte / Nte;
        v_acc[ep] = Ate / Nte;

        update_graph(t_loss, t_acc, v_loss, v_acc, ep + 1);
        printf("Epoch %3d: Tr(L=%.4f A=%.2f%%)  Te(L=%.4f A=%.2f%%)\n",
               ep + 1, t_loss[ep], t_acc[ep] * 100, v_loss[ep], v_acc[ep] * 100);

        net_save(net, SAVEFILE);
    }

    free(Xb_aug);
    free(t_loss);
    free(t_acc);
    free(v_loss);
    free(v_acc);
}

// 1) Create a brand-new network by asking for layer sizes, then reallocate buffers
void create_new_network_interface(
    Net *net, int *L, int **sizes,
    float ***A, float ***dZ,
    float ***dW, float ***db,
    float ***dA)
{
    printf("Enter number of layers: ");
    scanf("%d", L);
    *sizes = malloc((*L) * sizeof(int));
    for (int i = 0; i < *L; i++)
    {
        if (i == 0)
        {
            (*sizes)[i] = IMG_W * IMG_H;
            printf("Layer %d size set to %d (input)\n", i, (*sizes)[i]);
        }
        else
        {
            printf("Enter size of layer %d: ", i);
            scanf("%d", &(*sizes)[i]);
        }
    }
    net_init(net, *L, *sizes);
    // free old buffers if any
    if (*A)
        free_buffers(*L, *A, *dZ, *dW, *db, *dA);
    allocate_buffers(*L, *sizes, A, dZ, dW, db, dA);
    printf("New network created with %d layers.\n", *L);
}

// 2) Load a network from disk, then reallocate buffers
static void load_network_interface(
    Net *net,
    int *L,
    int **sizes,
    float ***A,
    float ***dZ,
    float ***dW,
    float ***db,
    float ***dA)
{
    char filename[256];
    printf("Enter filename to load: ");
    scanf("%255s", filename);

    if (!net_load(net, filename))
    {
        printf("❌ Failed to load '%s'\n", filename);
        return;
    }

    // free any existing buffers and sizes
    if (*A)
    {
        free_buffers(*L, *A, *dZ, *dW, *db, *dA);
        free(*sizes);
    }

    // update layer count and sizes from the loaded net
    *L = net->L;
    *sizes = malloc((*L) * sizeof(int));
    memcpy(*sizes, net->sz, (*L) * sizeof(int));

    // reallocate buffers for forward/backprop
    allocate_buffers(*L, *sizes, A, dZ, dW, db, dA);

    printf("✅ Loaded network '%s' with %d layers: [", filename, *L);
    for (int i = 0; i < *L; i++)
    {
        printf("%d%s", (*sizes)[i], i + 1 < *L ? ", " : "");
    }
    printf("]\n");
}

// 3) Save the current network to disk
void save_network_interface(Net *net)
{
    char filename[256];
    printf("Enter filename to save: ");
    scanf("%255s", filename);
    net_save(net, filename);
    printf("Network saved to '%s'.\n", filename);
}

// 4) Evaluate the network on the test set
void test_network_interface(
    Net *net,
    float *Xte, float *Yte, int Nte,
    float **A)
{
    double loss, acc;
    eval_epoch(net, Xte, Yte, Nte, A, &loss, &acc, 512);
    printf("Test results — loss: %.4f, accuracy: %.2f%%\n",
           loss / Nte, acc * 100.0 / Nte);
}

int main(int argc, char *argv[])
{
    srand((unsigned)time(NULL));
    setup_signal_handlers();

    // immediate-exit point
    if (setjmp(jump_env))
    {
        shutdown_graph();
        return 0;
    }

    init_graph();
    pthread_t evt_thread;
    pthread_create(&evt_thread, NULL, sdl_event_thread, NULL);
    pthread_detach(evt_thread);

    Net net;
    int L = 0, *sizes = NULL;
    float *Xtr = NULL, *Ytr = NULL, *Xte = NULL, *Yte = NULL;
    int Ntr = 0, Nte = 0;
    float **A = NULL, **dZ = NULL, **dW = NULL, **db = NULL, **dA = NULL;
    int choice;

    while (1)
    {
        printf("\n------------------------- Neural Network Interface ----------------------\n");
        printf(" Options:\n");
        printf(" 1) Train\n");
        printf(" 2) Save\n");
        printf(" 3) Load\n");
        printf(" 4) Test\n");
        printf(" 5) Load dataset\n");
        printf(" 6) Create new Network\n");
        printf(" 0) Exit\n");
        printf(" Choose one: ");
        if (scanf("%d", &choice) != 1)
        {
            getchar();
            continue;
        }

        switch (choice)
        {
        case 1:
            if (!A || !Xtr)
            {
                printf("Load network and dataset first.\n");
            }
            else
            {
                train_loop(&net, Xtr, Ytr, Ntr, Xte, Yte, Nte,
                           A, dZ, dW, db, dA);
            }
            break;
        case 2:
            if (!A)
                printf("No network loaded.\n");
            else
                save_network_interface(&net);
            break;
        case 3:
            load_network_interface(&net, &L, &sizes,
                                   &A, &dZ, &dW, &db, &dA);
            break;
        case 4:
            if (!A || !Xte)
            {
                printf("Load network and dataset first.\n");
            }
            else
            {
                test_network_interface(&net, Xte, Yte, Nte, A);
                shutdown_graph();
            }
            break;
        case 5:
            load_and_split_dataset(&Xtr, &Ytr, &Ntr, &Xte, &Yte, &Nte);
            printf("Dataset loaded: %d train / %d test samples.\n", Ntr, Nte);
            break;
        case 6:
            create_new_network_interface(&net, &L, &sizes,
                                         &A, &dZ, &dW, &db, &dA);
            break;
        case 0:
            goto cleanup;
        default:
            printf("Invalid option.\n");
        }
    }

cleanup:
    if (A)
    {
        free_buffers(L, A, dZ, dW, db, dA);
        net_free(&net);
    }
    return 0;
}