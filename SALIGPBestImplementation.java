import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.*;

/**
 * SALIGPBestImplementation
 * ------------------------------------------------------------
 * A runnable, non-hardcoded Java implementation inspired by the SALIGP paper.
 *
 * What it implements:
 * 1. Synthetic cloud dataset generation for 100, 200, 300, 400, and 500 files.
 * 2. Mixed file contents: text-like records, image-like bytes, csv-like rows, and metadata bytes.
 * 3. Duplicate and near-duplicate generation with ground-truth labels.
 * 4. Bloom Filter based fast chunk membership checks.
 * 5. Feature extraction: chunk similarity, Bloom hit ratio, byte histogram similarity, size similarity,
 *    full-hash match, and encrypted-hash evidence.
 * 6. Improved Genetic Programming style optimizer:
 *      - evolves weighted scoring expressions,
 *      - evolves decision threshold,
 *      - maximizes F1 score as fitness.
 * 7. Table-5-style experiment output for 100, 200, 300, 400, and 500 files.
 *
 * Important:
 * - This code DOES NOT hardcode the paper's Table 5 numbers.
 * - The output changes if the generated dataset or random seed changes.
 * - The paper's exact Table 5 cannot be guaranteed without the authors' original dataset and source code.
 *
 * Compile:
 *   javac SALIGPBestImplementation.java
 *
 * Run:
 *   java SALIGPBestImplementation
 *
 * Optional:
 *   java SALIGPBestImplementation output/result.csv
 */
public class SALIGPBestImplementation {
    static final int CHUNK_SIZE = 4096;
    static final int BLOOM_SIZE = 1_000_003;
    static final int BLOOM_HASHES = 5;
    static final int[] EXPERIMENT_FILE_COUNTS = {100, 200, 300, 400, 500};
    static final long SEED = 20260612L;

    static final SecureRandom RANDOM = new SecureRandom(longToBytes(SEED));

    static byte[] longToBytes(long x) {
        byte[] b = new byte[8];
        for (int i = 7; i >= 0; i--) {
            b[i] = (byte) (x & 0xff);
            x >>= 8;
        }
        return b;
    }

    static class CloudFile {
        int id;
        int sourceId;
        boolean duplicate;
        byte[] data;
        String name;

        CloudFile(int id, int sourceId, boolean duplicate, byte[] data) {
            this.id = id;
            this.sourceId = sourceId;
            this.duplicate = duplicate;
            this.data = data;
            this.name = String.format(Locale.US, "file_%04d.bin", id);
        }
    }

    static class Profile {
        CloudFile file;
        String fullHash;
        String encryptedHash;
        Set<String> chunks = new HashSet<String>();
        int[] histogram = new int[16];
        int size;
    }

    static class PairSample {
        Profile a;
        Profile b;
        boolean duplicate;
        double[] features;

        PairSample(Profile a, Profile b, boolean duplicate, double[] features) {
            this.a = a;
            this.b = b;
            this.duplicate = duplicate;
            this.features = features;
        }
    }

    static class Individual {
        double[] weights;
        double threshold;
        double fitness;
        double precision;
        double recall;
        double f1;

        Individual(int features) {
            weights = new double[features];
        }

        Individual copy() {
            Individual n = new Individual(weights.length);
            System.arraycopy(weights, 0, n.weights, 0, weights.length);
            n.threshold = threshold;
            n.fitness = fitness;
            n.precision = precision;
            n.recall = recall;
            n.f1 = f1;
            return n;
        }
    }

    static class ExperimentResult {
        int files;
        int trueDuplicatePairs;
        int detectedDuplicatePairs;
        int correctDuplicatePairs;
        double precision;
        double recall;
        double f1;
        double detectionPercent;
        double threshold;
        long timeMs;
    }

    static class BloomFilter {
        BitSet bits = new BitSet(BLOOM_SIZE);

        int index(String value, int salt) throws Exception {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            md.update(value.getBytes(StandardCharsets.UTF_8));
            md.update((byte) salt);
            byte[] d = md.digest();
            int v = 0;
            for (int i = 0; i < 4; i++) v = (v << 8) | (d[i] & 0xff);
            int m = v % BLOOM_SIZE;
            return m < 0 ? m + BLOOM_SIZE : m;
        }

        void add(String value) throws Exception {
            for (int i = 0; i < BLOOM_HASHES; i++) bits.set(index(value, i));
        }

        boolean mightContain(String value) throws Exception {
            for (int i = 0; i < BLOOM_HASHES; i++) {
                if (!bits.get(index(value, i))) return false;
            }
            return true;
        }
    }

    static String sha256(byte[] data) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        byte[] d = md.digest(data);
        StringBuilder sb = new StringBuilder();
        for (byte b : d) sb.append(String.format("%02x", b));
        return sb.toString();
    }

    static byte[] aesEncrypt(byte[] data) throws Exception {
        byte[] key = Arrays.copyOf(MessageDigest.getInstance("SHA-256").digest("SALIGP_KEY".getBytes(StandardCharsets.UTF_8)), 16);
        SecretKeySpec spec = new SecretKeySpec(key, "AES");
        Cipher cipher = Cipher.getInstance("AES/ECB/PKCS5Padding");
        cipher.init(Cipher.ENCRYPT_MODE, spec);
        return cipher.doFinal(data);
    }

    static byte[] makeMixedContent(int id, int sizeBytes) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        String header = "SALIGP cloud object " + id + "\n" +
                "type=text+image+csv+metadata\n" +
                "owner=UID" + (1000 + id) + "\n" +
                "role=level" + (id % 4) + "\n" +
                "records=secure deduplication active learning bloom filter genetic programming\n";
        write(out, header.getBytes(StandardCharsets.UTF_8));

        for (int i = 0; i < 200 && out.size() < sizeBytes; i++) {
            String row = "row," + id + "," + i + "," + ((id * 31 + i * 17) % 997) + ",cloud-storage-dedup\n";
            write(out, row.getBytes(StandardCharsets.UTF_8));
        }

        byte[] imageLike = new byte[4096];
        for (int i = 0; i < imageLike.length; i++) imageLike[i] = (byte) ((id * 13 + i * 7 + 91) & 0xff);
        while (out.size() + imageLike.length < sizeBytes) write(out, imageLike);

        while (out.size() < sizeBytes) out.write((byte) ((id * 19 + out.size() * 5) & 0xff));
        byte[] data = out.toByteArray();
        return Arrays.copyOf(data, sizeBytes);
    }

    static void write(ByteArrayOutputStream out, byte[] data) {
        try {
            out.write(data);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    static byte[] mutate(byte[] source, int changes) {
        byte[] data = Arrays.copyOf(source, source.length);
        Random r = new Random(SEED + source.length + changes);
        for (int i = 0; i < changes; i++) {
            int pos = r.nextInt(data.length);
            data[pos] = (byte) (data[pos] ^ (1 + r.nextInt(127)));
        }
        return data;
    }

    static List<CloudFile> generateDataset(int count) {
        List<CloudFile> files = new ArrayList<CloudFile>();
        int baseCount = Math.max(10, (int) (count * 0.65));
        int minSize = 80 * 1024;
        int maxSize = 500 * 1024;

        for (int i = 0; i < baseCount; i++) {
            int size = minSize + (int) ((i * 7919L + count * 101L) % (maxSize - minSize));
            files.add(new CloudFile(i, i, false, makeMixedContent(i, size)));
        }

        for (int i = baseCount; i < count; i++) {
            int src = (int) ((i * 37L + count) % baseCount);
            CloudFile original = files.get(src);
            boolean exact = (i % 3 == 0);
            byte[] data = exact ? Arrays.copyOf(original.data, original.data.length)
                    : mutate(original.data, Math.max(8, original.data.length / 200));
            files.add(new CloudFile(i, src, true, data));
        }

        Collections.shuffle(files, new Random(SEED + count));
        return files;
    }

    static Profile profile(CloudFile f) throws Exception {
        Profile p = new Profile();
        p.file = f;
        p.size = f.data.length;
        p.fullHash = sha256(f.data);
        p.encryptedHash = sha256(aesEncrypt(f.data));
        for (int i = 0; i < f.data.length; i += CHUNK_SIZE) {
            int end = Math.min(i + CHUNK_SIZE, f.data.length);
            byte[] chunk = Arrays.copyOfRange(f.data, i, end);
            p.chunks.add(sha256(chunk));
        }
        for (byte b : f.data) p.histogram[(b & 0xff) / 16]++;
        return p;
    }

    static double jaccard(Set<String> a, Set<String> b) {
        Set<String> inter = new HashSet<String>(a);
        inter.retainAll(b);
        Set<String> union = new HashSet<String>(a);
        union.addAll(b);
        if (union.isEmpty()) return 1.0;
        return inter.size() / (double) union.size();
    }

    static double histSimilarity(Profile a, Profile b) {
        double min = 0, max = 0;
        for (int i = 0; i < 16; i++) {
            min += Math.min(a.histogram[i], b.histogram[i]);
            max += Math.max(a.histogram[i], b.histogram[i]);
        }
        return max == 0 ? 0 : min / max;
    }

    static double sizeSimilarity(Profile a, Profile b) {
        int max = Math.max(a.size, b.size);
        int diff = Math.abs(a.size - b.size);
        return max == 0 ? 1.0 : 1.0 - (diff / (double) max);
    }

    static double bloomHitRatio(Profile candidate, BloomFilter bf) throws Exception {
        int hit = 0;
        for (String c : candidate.chunks) if (bf.mightContain(c)) hit++;
        return candidate.chunks.isEmpty() ? 0 : hit / (double) candidate.chunks.size();
    }

    static double[] features(Profile a, Profile b, BloomFilter bf) throws Exception {
        return new double[]{
                jaccard(a.chunks, b.chunks),
                histSimilarity(a, b),
                sizeSimilarity(a, b),
                a.fullHash.equals(b.fullHash) ? 1.0 : 0.0,
                a.encryptedHash.equals(b.encryptedHash) ? 1.0 : 0.0,
                bloomHitRatio(a, bf)
        };
    }

    static List<PairSample> makePairs(List<Profile> profiles) throws Exception {
        BloomFilter bf = new BloomFilter();
        for (Profile p : profiles) for (String c : p.chunks) bf.add(c);

        List<PairSample> pairs = new ArrayList<PairSample>();
        Map<Integer, List<Profile>> bySource = new HashMap<Integer, List<Profile>>();
        for (Profile p : profiles) bySource.computeIfAbsent(p.file.sourceId, k -> new ArrayList<Profile>()).add(p);

        // Positive pairs: files sharing same source id.
        for (List<Profile> group : bySource.values()) {
            for (int i = 0; i < group.size(); i++) {
                for (int j = i + 1; j < group.size(); j++) {
                    Profile a = group.get(i), b = group.get(j);
                    pairs.add(new PairSample(a, b, true, features(a, b, bf)));
                }
            }
        }

        // Negative pairs: deterministic sampling from different source ids.
        int targetNegatives = Math.max(1, pairs.size() * 2);
        Random r = new Random(SEED + profiles.size());
        int attempts = 0;
        while (targetNegatives > 0 && attempts < profiles.size() * profiles.size() * 5) {
            attempts++;
            Profile a = profiles.get(r.nextInt(profiles.size()));
            Profile b = profiles.get(r.nextInt(profiles.size()));
            if (a == b || a.file.sourceId == b.file.sourceId) continue;
            pairs.add(new PairSample(a, b, false, features(a, b, bf)));
            targetNegatives--;
        }
        Collections.shuffle(pairs, new Random(SEED + 99 + profiles.size()));
        return pairs;
    }

    static double score(Individual ind, double[] f) {
        double s = 0;
        double sum = 0;
        for (int i = 0; i < ind.weights.length; i++) {
            s += ind.weights[i] * f[i];
            sum += Math.abs(ind.weights[i]);
        }
        return sum == 0 ? 0 : s / sum;
    }

    static void evaluateIndividual(Individual ind, List<PairSample> samples) {
        int tp = 0, fp = 0, fn = 0;
        for (PairSample p : samples) {
            boolean predicted = score(ind, p.features) >= ind.threshold;
            if (predicted && p.duplicate) tp++;
            else if (predicted) fp++;
            else if (p.duplicate) fn++;
        }
        ind.precision = tp + fp == 0 ? 0 : tp / (double) (tp + fp);
        ind.recall = tp + fn == 0 ? 0 : tp / (double) (tp + fn);
        ind.f1 = ind.precision + ind.recall == 0 ? 0 : 2 * ind.precision * ind.recall / (ind.precision + ind.recall);
        ind.fitness = ind.f1;
    }

    static Individual randomIndividual(Random r, int featureCount) {
        Individual ind = new Individual(featureCount);
        for (int i = 0; i < featureCount; i++) ind.weights[i] = 0.1 + r.nextDouble();
        ind.threshold = 0.35 + r.nextDouble() * 0.45;
        return ind;
    }

    static Individual crossover(Individual a, Individual b, Random r) {
        Individual c = new Individual(a.weights.length);
        for (int i = 0; i < c.weights.length; i++) c.weights[i] = r.nextBoolean() ? a.weights[i] : b.weights[i];
        c.threshold = (a.threshold + b.threshold) / 2.0;
        return c;
    }

    static void mutate(Individual ind, Random r) {
        for (int i = 0; i < ind.weights.length; i++) {
            if (r.nextDouble() < 0.25) ind.weights[i] += r.nextGaussian() * 0.08;
            if (ind.weights[i] < 0.01) ind.weights[i] = 0.01;
            if (ind.weights[i] > 2.0) ind.weights[i] = 2.0;
        }
        if (r.nextDouble() < 0.35) ind.threshold += r.nextGaussian() * 0.04;
        if (ind.threshold < 0.20) ind.threshold = 0.20;
        if (ind.threshold > 0.95) ind.threshold = 0.95;
    }

    static Individual trainIGP(List<PairSample> train) {
        Random r = new Random(SEED + train.size());
        int populationSize = 60;
        int generations = 60;
        int featureCount = train.get(0).features.length;
        List<Individual> pop = new ArrayList<Individual>();
        for (int i = 0; i < populationSize; i++) pop.add(randomIndividual(r, featureCount));

        for (int g = 0; g < generations; g++) {
            for (Individual ind : pop) evaluateIndividual(ind, train);
            Collections.sort(pop, (x, y) -> Double.compare(y.fitness, x.fitness));
            List<Individual> next = new ArrayList<Individual>();
            for (int i = 0; i < 8; i++) next.add(pop.get(i).copy());
            while (next.size() < populationSize) {
                Individual p1 = pop.get(r.nextInt(20));
                Individual p2 = pop.get(r.nextInt(20));
                Individual child = crossover(p1, p2, r);
                mutate(child, r);
                next.add(child);
            }
            pop = next;
        }
        for (Individual ind : pop) evaluateIndividual(ind, train);
        Collections.sort(pop, (x, y) -> Double.compare(y.fitness, x.fitness));
        return pop.get(0).copy();
    }

    static ExperimentResult runExperiment(int count) throws Exception {
        long start = System.currentTimeMillis();
        List<CloudFile> files = generateDataset(count);
        List<Profile> profiles = new ArrayList<Profile>();
        for (CloudFile f : files) profiles.add(profile(f));
        List<PairSample> pairs = makePairs(profiles);
        int split = Math.max(1, (int) (pairs.size() * 0.70));
        List<PairSample> train = pairs.subList(0, split);
        List<PairSample> test = pairs.subList(split, pairs.size());
        if (test.isEmpty()) test = train;

        Individual best = trainIGP(train);
        int tp = 0, fp = 0, fn = 0, trueDup = 0, detected = 0;
        for (PairSample p : test) {
            if (p.duplicate) trueDup++;
            boolean pred = score(best, p.features) >= best.threshold;
            if (pred) detected++;
            if (pred && p.duplicate) tp++;
            else if (pred) fp++;
            else if (p.duplicate) fn++;
        }
        ExperimentResult r = new ExperimentResult();
        r.files = count;
        r.trueDuplicatePairs = trueDup;
        r.detectedDuplicatePairs = detected;
        r.correctDuplicatePairs = tp;
        r.precision = tp + fp == 0 ? 0 : tp / (double) (tp + fp);
        r.recall = tp + fn == 0 ? 0 : tp / (double) (tp + fn);
        r.f1 = r.precision + r.recall == 0 ? 0 : 2 * r.precision * r.recall / (r.precision + r.recall);
        r.detectionPercent = r.recall * 100.0;
        r.threshold = best.threshold;
        r.timeMs = System.currentTimeMillis() - start;
        return r;
    }

    static void writeResults(List<ExperimentResult> results, Path out) throws Exception {
        Files.createDirectories(out.toAbsolutePath().getParent());
        BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8);
        w.write("files,true_duplicate_pairs,detected_duplicate_pairs,correct_duplicate_pairs,precision,recall,f1,saligp_detection_percent,threshold,time_ms\n");
        for (ExperimentResult r : results) {
            w.write(String.format(Locale.US,
                    "%d,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.4f,%d\n",
                    r.files, r.trueDuplicatePairs, r.detectedDuplicatePairs, r.correctDuplicatePairs,
                    r.precision, r.recall, r.f1, r.detectionPercent, r.threshold, r.timeMs));
        }
        w.close();
    }

    public static void main(String[] args) throws Exception {
        Path output = args.length > 0 ? Paths.get(args[0]) : Paths.get("output/saligp_table5_style_result.csv");
        List<ExperimentResult> results = new ArrayList<ExperimentResult>();
        System.out.println("Running SALIGP Table-5-style experiment...");
        System.out.println("This computes values from generated duplicate/near-duplicate datasets; it does not hardcode Table 5.");
        for (int count : EXPERIMENT_FILE_COUNTS) {
            ExperimentResult r = runExperiment(count);
            results.add(r);
            System.out.printf(Locale.US,
                    "Files=%d | Detection=%.2f%% | Precision=%.4f | Recall=%.4f | F1=%.4f | Time=%d ms%n",
                    r.files, r.detectionPercent, r.precision, r.recall, r.f1, r.timeMs);
        }
        writeResults(results, output);
        System.out.println("CSV saved to: " + output.toAbsolutePath());
    }
}
