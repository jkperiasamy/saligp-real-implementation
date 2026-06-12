import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * SALIGP Table-5-style implementation using FILE COUNTS as input.
 *
 * This program follows the Table 5 experiment format:
 *   100 files, 200 files, 300 files, 400 files, 500 files
 *
 * It does NOT use 100KB, 200KB, etc. as the experiment input.
 * It also does NOT hardcode the paper's Table 5 output values.
 *
 * For each file-count setting, it generates a mixed dataset containing:
 *   - unique files
 *   - exact duplicates
 *   - near duplicates
 *   - noisy files
 *
 * Then it runs a SALIGP-inspired duplicate detector:
 *   - chunk hashing
 *   - Bloom filter evidence
 *   - histogram similarity
 *   - IGP-style threshold search using F1 fitness
 *
 * Compile:
 *   javac SALIGPTable5FilesImplementation.java
 *
 * Run:
 *   java SALIGPTable5FilesImplementation output/table5_output.csv
 */
public class SALIGPTable5FilesImplementation {
    static final int[] FILE_COUNTS = {100, 200, 300, 400, 500};
    static final int CHUNK_SIZE = 512;
    static final int BLOOM_SIZE = 200003;
    static final int BLOOM_HASHES = 4;
    static final long SEED = 20260612L;

    static class CloudFile {
        String id;
        String familyId;
        boolean duplicate;
        byte[] content;
        Set<String> chunks = new HashSet<String>();
        int[] histogram = new int[16];
    }

    static class PairSample {
        CloudFile a;
        CloudFile b;
        boolean actualDuplicate;
        double chunkSimilarity;
        double histogramSimilarity;
        double bloomEvidence;
        double score;
    }

    static class Result {
        int files;
        int trueDuplicatePairs;
        int detectedDuplicatePairs;
        int correctDuplicatePairs;
        double precision;
        double recall;
        double f1;
        double detectionPercent;
        double bestThreshold;
        long timeMs;
    }

    static class BloomFilter {
        BitSet bits = new BitSet(BLOOM_SIZE);

        int index(byte[] digest, int salt) {
            int value = 0;
            for (int i = 0; i < 4; i++) {
                value = (value << 8) | (digest[(salt + i) % digest.length] & 0xff);
            }
            int mod = value % BLOOM_SIZE;
            return mod < 0 ? mod + BLOOM_SIZE : mod;
        }

        void add(String text) throws Exception {
            byte[] digest = sha256(text.getBytes(StandardCharsets.UTF_8));
            for (int i = 0; i < BLOOM_HASHES; i++) bits.set(index(digest, i * 5));
        }

        boolean mightContain(String text) throws Exception {
            byte[] digest = sha256(text.getBytes(StandardCharsets.UTF_8));
            for (int i = 0; i < BLOOM_HASHES; i++) {
                if (!bits.get(index(digest, i * 5))) return false;
            }
            return true;
        }
    }

    static byte[] sha256(byte[] input) throws Exception {
        return MessageDigest.getInstance("SHA-256").digest(input);
    }

    static String hex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) sb.append(String.format("%02x", b));
        return sb.toString();
    }

    static byte[] createBaseContent(Random random, int fileIndex) {
        String[] topics = {
            "cloud security deduplication storage encryption access control ",
            "machine learning bloom filter genetic programming similarity ",
            "network traffic authentication blockchain cloud server privacy ",
            "image metadata binary payload checksum chunk hash table ",
            "document record user role hierarchy private key ciphertext "
        };
        StringBuilder sb = new StringBuilder();
        int targetSize = 1800 + random.nextInt(2200);
        while (sb.length() < targetSize) {
            sb.append("FILE_").append(fileIndex).append('_');
            sb.append(topics[random.nextInt(topics.length)]);
            sb.append("value=").append(random.nextInt(100000)).append(';');
        }
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    static byte[] mutate(byte[] source, Random random, double changeRate) {
        byte[] copy = source.clone();
        int changes = Math.max(1, (int)(copy.length * changeRate));
        for (int i = 0; i < changes; i++) {
            int pos = random.nextInt(copy.length);
            copy[pos] = (byte)('A' + random.nextInt(26));
        }
        return copy;
    }

    static CloudFile makeFile(String id, String familyId, boolean duplicate, byte[] content) throws Exception {
        CloudFile f = new CloudFile();
        f.id = id;
        f.familyId = familyId;
        f.duplicate = duplicate;
        f.content = content;
        for (int i = 0; i < content.length; i += CHUNK_SIZE) {
            int end = Math.min(i + CHUNK_SIZE, content.length);
            byte[] chunk = new byte[end - i];
            System.arraycopy(content, i, chunk, 0, chunk.length);
            f.chunks.add(hex(sha256(chunk)));
        }
        for (byte b : content) f.histogram[(b & 0xff) / 16]++;
        return f;
    }

    static List<CloudFile> generateDataset(int count, Random random) throws Exception {
        List<CloudFile> files = new ArrayList<CloudFile>();
        int duplicateFamilies = Math.max(8, count / 10);
        int idCounter = 0;

        for (int i = 0; i < duplicateFamilies; i++) {
            byte[] base = createBaseContent(random, i);
            String family = "DUP_FAMILY_" + i;
            files.add(makeFile("file_" + (idCounter++), family, false, base));

            double noise = 0.01 + random.nextDouble() * 0.12;
            byte[] dup = random.nextBoolean() ? base.clone() : mutate(base, random, noise);
            files.add(makeFile("file_" + (idCounter++), family, true, dup));
        }

        while (files.size() < count) {
            byte[] unique = createBaseContent(random, idCounter + 10000);
            // Add some confusing but non-duplicate near-topic files.
            if (random.nextDouble() < 0.25 && !files.isEmpty()) {
                CloudFile source = files.get(random.nextInt(files.size()));
                unique = mutate(source.content, random, 0.35 + random.nextDouble() * 0.35);
            }
            files.add(makeFile("file_" + (idCounter++), "UNIQUE_" + idCounter, false, unique));
        }

        Collections.shuffle(files, random);
        return files;
    }

    static double jaccard(Set<String> a, Set<String> b) {
        Set<String> inter = new HashSet<String>(a);
        inter.retainAll(b);
        Set<String> union = new HashSet<String>(a);
        union.addAll(b);
        return union.isEmpty() ? 0.0 : inter.size() / (double) union.size();
    }

    static double histogramSimilarity(CloudFile a, CloudFile b) {
        double min = 0.0, max = 0.0;
        for (int i = 0; i < 16; i++) {
            min += Math.min(a.histogram[i], b.histogram[i]);
            max += Math.max(a.histogram[i], b.histogram[i]);
        }
        return max == 0.0 ? 0.0 : min / max;
    }

    static double bloomEvidence(CloudFile a, CloudFile b) throws Exception {
        BloomFilter bf = new BloomFilter();
        for (String h : a.chunks) bf.add(h);
        int hits = 0;
        for (String h : b.chunks) if (bf.mightContain(h)) hits++;
        return b.chunks.isEmpty() ? 0.0 : hits / (double) b.chunks.size();
    }

    static List<PairSample> buildCandidatePairs(List<CloudFile> files) throws Exception {
        List<PairSample> pairs = new ArrayList<PairSample>();
        Map<String, List<CloudFile>> byFamily = new HashMap<String, List<CloudFile>>();
        for (CloudFile f : files) {
            if (!byFamily.containsKey(f.familyId)) byFamily.put(f.familyId, new ArrayList<CloudFile>());
            byFamily.get(f.familyId).add(f);
        }

        // Positive pairs from same duplicate families.
        for (List<CloudFile> group : byFamily.values()) {
            if (group.size() < 2) continue;
            for (int i = 0; i < group.size(); i++) {
                for (int j = i + 1; j < group.size(); j++) {
                    pairs.add(makePair(group.get(i), group.get(j), true));
                }
            }
        }

        // Negative pairs: enough to challenge the detector.
        Random random = new Random(SEED + files.size());
        int negativeTarget = Math.max(pairs.size() * 4, files.size());
        Set<String> seen = new HashSet<String>();
        while (seen.size() < negativeTarget) {
            CloudFile a = files.get(random.nextInt(files.size()));
            CloudFile b = files.get(random.nextInt(files.size()));
            if (a == b) continue;
            if (a.familyId.equals(b.familyId)) continue;
            String key = a.id.compareTo(b.id) < 0 ? a.id + "|" + b.id : b.id + "|" + a.id;
            if (seen.add(key)) pairs.add(makePair(a, b, false));
        }
        return pairs;
    }

    static PairSample makePair(CloudFile a, CloudFile b, boolean actual) throws Exception {
        PairSample p = new PairSample();
        p.a = a;
        p.b = b;
        p.actualDuplicate = actual;
        p.chunkSimilarity = jaccard(a.chunks, b.chunks);
        p.histogramSimilarity = histogramSimilarity(a, b);
        p.bloomEvidence = bloomEvidence(a, b);
        p.score = igpScore(p.chunkSimilarity, p.histogramSimilarity, p.bloomEvidence);
        return p;
    }

    static double igpScore(double chunkSimilarity, double histogramSimilarity, double bloomEvidence) {
        // IGP-style expression: weighted evidence with a mild interaction term.
        double score = 0.52 * chunkSimilarity + 0.23 * histogramSimilarity + 0.20 * bloomEvidence;
        score += 0.05 * Math.sqrt(Math.max(0.0, chunkSimilarity * histogramSimilarity));
        return Math.max(0.0, Math.min(1.0, score));
    }

    static Result evaluate(int fileCount) throws Exception {
        long start = System.currentTimeMillis();
        Random random = new Random(SEED + fileCount);
        List<CloudFile> files = generateDataset(fileCount, random);
        List<PairSample> pairs = buildCandidatePairs(files);

        double bestThreshold = 0.50;
        double bestF1 = -1.0;
        for (double t = 0.30; t <= 0.90; t += 0.005) {
            double f1 = computeMetrics(fileCount, pairs, t, 0).f1;
            if (f1 > bestF1) {
                bestF1 = f1;
                bestThreshold = t;
            }
        }

        Result r = computeMetrics(fileCount, pairs, bestThreshold, System.currentTimeMillis() - start);
        r.bestThreshold = bestThreshold;
        return r;
    }

    static Result computeMetrics(int files, List<PairSample> pairs, double threshold, long timeMs) {
        int truePairs = 0, detected = 0, correct = 0;
        for (PairSample p : pairs) {
            if (p.actualDuplicate) truePairs++;
            boolean predicted = p.score >= threshold;
            if (predicted) detected++;
            if (predicted && p.actualDuplicate) correct++;
        }
        Result r = new Result();
        r.files = files;
        r.trueDuplicatePairs = truePairs;
        r.detectedDuplicatePairs = detected;
        r.correctDuplicatePairs = correct;
        r.precision = detected == 0 ? 0.0 : correct / (double) detected;
        r.recall = truePairs == 0 ? 0.0 : correct / (double) truePairs;
        r.f1 = (r.precision + r.recall) == 0.0 ? 0.0 : (2 * r.precision * r.recall) / (r.precision + r.recall);
        r.detectionPercent = r.recall * 100.0;
        r.bestThreshold = threshold;
        r.timeMs = timeMs;
        return r;
    }

    static void writeCsv(List<Result> results, Path output) throws Exception {
        Path parent = output.getParent();
        if (parent != null) Files.createDirectories(parent);
        BufferedWriter w = Files.newBufferedWriter(output);
        w.write("files,true_duplicate_pairs,detected_duplicate_pairs,correct_duplicate_pairs,precision,recall,f1,detection_percent,best_threshold,time_ms\n");
        for (Result r : results) {
            w.write(String.format(Locale.US, "%d,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.4f,%d\n",
                    r.files, r.trueDuplicatePairs, r.detectedDuplicatePairs, r.correctDuplicatePairs,
                    r.precision, r.recall, r.f1, r.detectionPercent, r.bestThreshold, r.timeMs));
        }
        w.close();
    }

    public static void main(String[] args) throws Exception {
        Path output = args.length > 0 ? Paths.get(args[0]) : Paths.get("output/table5_files_output.csv");
        List<Result> results = new ArrayList<Result>();
        System.out.println("Running Table 5 file-count experiment: 100, 200, 300, 400, 500 files");
        for (int count : FILE_COUNTS) {
            Result r = evaluate(count);
            results.add(r);
            System.out.println(String.format(Locale.US,
                    "Files=%d | Detection=%.2f%% | Precision=%.4f | Recall=%.4f | F1=%.4f | Threshold=%.4f | Time=%d ms",
                    r.files, r.detectionPercent, r.precision, r.recall, r.f1, r.bestThreshold, r.timeMs));
        }
        writeCsv(results, output);
        System.out.println("CSV saved to: " + output.toAbsolutePath());
    }
}
