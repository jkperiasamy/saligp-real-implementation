import java.io.*;
import java.nio.file.*;
import java.security.*;
import java.util.*;

/**
 * SALIGPRealImplementation
 * ------------------------------------------------------------
 * This is a runnable implementation inspired by the paper's SALIGP flow:
 * 1. split files into chunks,
 * 2. create hash-based features,
 * 3. use a Bloom filter for fast duplicate checks,
 * 4. use an IGP-style weighted fitness score for duplicate detection,
 * 5. report duplicate detection percentage for 100KB-500KB files.
 *
 * This version DOES NOT hardcode Table 5 output values. The output is computed
 * from the actual input file contents.
 *
 * Compile:
 *   javac SALIGPRealImplementation.java
 *
 * Run from src folder:
 *   java SALIGPRealImplementation ../input_files ../output/result.csv
 */
public class SALIGPRealImplementation {
    static final int CHUNK_SIZE = 4096;
    static final int BLOOM_SIZE = 1_000_003;
    static final int HASH_FUNCTIONS = 5;
    static final double DUPLICATE_THRESHOLD = 0.72;

    static class BloomFilter {
        private final BitSet bits;
        private final int size;
        private final int functions;

        BloomFilter(int size, int functions) {
            this.size = size;
            this.functions = functions;
            this.bits = new BitSet(size);
        }

        private int index(byte[] digest, int salt) {
            int value = 0;
            for (int i = 0; i < 4; i++) {
                value = (value << 8) | (digest[(salt + i) % digest.length] & 0xff);
            }
            int mod = value % size;
            if (mod < 0) mod += size;
            return mod;
        }

        void add(byte[] digest) {
            for (int i = 0; i < functions; i++) bits.set(index(digest, i * 3));
        }

        boolean mightContain(byte[] digest) {
            for (int i = 0; i < functions; i++) {
                if (!bits.get(index(digest, i * 3))) return false;
            }
            return true;
        }
    }

    static class FileProfile {
        String name;
        long sizeBytes;
        String sha256;
        List<String> chunkHashes = new ArrayList<String>();
        Set<String> chunkHashSet = new HashSet<String>();
        Map<String, Integer> byteHistogram = new HashMap<String, Integer>();
        byte[] fullDigest;
    }

    static class Result {
        String fileName;
        long sizeKB;
        int totalChunks;
        int duplicateChunks;
        double bloomHitPercent;
        double chunkSimilarity;
        double histogramSimilarity;
        double igpScore;
        String decision;
        long timeMs;
    }

    static MessageDigest newDigest() throws Exception {
        return MessageDigest.getInstance("SHA-256");
    }

    static String hex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) sb.append(String.format("%02x", b));
        return sb.toString();
    }

    static FileProfile profile(Path path) throws Exception {
        FileProfile fp = new FileProfile();
        fp.name = path.getFileName().toString();
        fp.sizeBytes = Files.size(path);
        byte[] data = Files.readAllBytes(path);

        MessageDigest full = newDigest();
        fp.fullDigest = full.digest(data);
        fp.sha256 = hex(fp.fullDigest);

        for (int i = 0; i < data.length; i += CHUNK_SIZE) {
            int end = Math.min(i + CHUNK_SIZE, data.length);
            byte[] chunk = Arrays.copyOfRange(data, i, end);
            String h = hex(newDigest().digest(chunk));
            fp.chunkHashes.add(h);
            fp.chunkHashSet.add(h);
        }

        int[] hist = new int[16];
        for (byte b : data) hist[(b & 0xff) / 16]++;
        for (int i = 0; i < hist.length; i++) fp.byteHistogram.put(String.valueOf(i), hist[i]);
        return fp;
    }

    static double jaccard(Set<String> a, Set<String> b) {
        if (a.isEmpty() && b.isEmpty()) return 1.0;
        Set<String> inter = new HashSet<String>(a);
        inter.retainAll(b);
        Set<String> union = new HashSet<String>(a);
        union.addAll(b);
        return inter.size() / (double) union.size();
    }

    static double histogramSimilarity(FileProfile a, FileProfile b) {
        double minSum = 0, maxSum = 0;
        for (int i = 0; i < 16; i++) {
            int x = a.byteHistogram.get(String.valueOf(i));
            int y = b.byteHistogram.get(String.valueOf(i));
            minSum += Math.min(x, y);
            maxSum += Math.max(x, y);
        }
        if (maxSum == 0) return 0;
        return minSum / maxSum;
    }

    static double igpScore(double chunkSim, double histSim, double bloomHit) {
        // IGP-style weighted expression evolved for deduplication-like scoring.
        // Uses actual computed evidence values; no fixed Table 5 values.
        double evidence1 = 0.55 * chunkSim;
        double evidence2 = 0.30 * histSim;
        double evidence3 = 0.15 * bloomHit;
        return evidence1 + evidence2 + evidence3;
    }

    static Result evaluate(FileProfile candidate, List<FileProfile> cloud, BloomFilter bloom) throws Exception {
        long start = System.currentTimeMillis();
        Result r = new Result();
        r.fileName = candidate.name;
        r.sizeKB = candidate.sizeBytes / 1024;
        r.totalChunks = candidate.chunkHashes.size();

        int hits = 0;
        for (String h : candidate.chunkHashes) {
            if (bloom.mightContain(h.getBytes("UTF-8"))) hits++;
        }
        r.duplicateChunks = hits;
        r.bloomHitPercent = hits / (double) Math.max(1, r.totalChunks);

        double bestChunk = 0.0;
        double bestHist = 0.0;
        for (FileProfile stored : cloud) {
            bestChunk = Math.max(bestChunk, jaccard(candidate.chunkHashSet, stored.chunkHashSet));
            bestHist = Math.max(bestHist, histogramSimilarity(candidate, stored));
        }
        r.chunkSimilarity = bestChunk;
        r.histogramSimilarity = bestHist;
        r.igpScore = igpScore(r.chunkSimilarity, r.histogramSimilarity, r.bloomHitPercent);
        r.decision = r.igpScore >= DUPLICATE_THRESHOLD ? "Duplicate" : "Unique";
        r.timeMs = System.currentTimeMillis() - start;
        return r;
    }

    static void addToCloud(FileProfile fp, List<FileProfile> cloud, BloomFilter bloom) throws Exception {
        cloud.add(fp);
        for (String h : fp.chunkHashes) bloom.add(h.getBytes("UTF-8"));
    }

    static void writeCsv(List<Result> results, Path output) throws Exception {
        BufferedWriter w = Files.newBufferedWriter(output);
        w.write("file,size_kb,total_chunks,duplicate_chunks,bloom_hit_percent,chunk_similarity,histogram_similarity,igp_score,decision,time_ms\n");
        for (Result r : results) {
            w.write(r.fileName + "," + r.sizeKB + "," + r.totalChunks + "," + r.duplicateChunks + "," +
                    String.format(Locale.US, "%.2f", r.bloomHitPercent * 100) + "," +
                    String.format(Locale.US, "%.4f", r.chunkSimilarity) + "," +
                    String.format(Locale.US, "%.4f", r.histogramSimilarity) + "," +
                    String.format(Locale.US, "%.4f", r.igpScore) + "," + r.decision + "," + r.timeMs + "\n");
        }
        w.close();
    }

    public static void main(String[] args) throws Exception {
        Path inputDir = args.length > 0 ? Paths.get(args[0]) : Paths.get("../input_files");
        Path outputCsv = args.length > 1 ? Paths.get(args[1]) : Paths.get("../output/result.csv");
        Files.createDirectories(outputCsv.getParent());

        List<Path> files = new ArrayList<Path>();
        DirectoryStream<Path> stream = Files.newDirectoryStream(inputDir, "*.bin");
        for (Path p : stream) files.add(p);
        Collections.sort(files);

        BloomFilter bloom = new BloomFilter(BLOOM_SIZE, HASH_FUNCTIONS);
        List<FileProfile> cloudStorage = new ArrayList<FileProfile>();
        List<Result> results = new ArrayList<Result>();

        // Seed cloud with reference files. These are not outputs; they simulate files already in cloud storage.
        for (Path p : files) {
            if (p.getFileName().toString().startsWith("cloud_seed")) {
                addToCloud(profile(p), cloudStorage, bloom);
            }
        }

        for (Path p : files) {
            if (!p.getFileName().toString().startsWith("input_")) continue;
            FileProfile fp = profile(p);
            Result r = evaluate(fp, cloudStorage, bloom);
            results.add(r);
            if ("Unique".equals(r.decision)) addToCloud(fp, cloudStorage, bloom);
        }

        writeCsv(results, outputCsv);
        for (Result r : results) {
            System.out.println(r.fileName + " => " + r.decision + " | IGP score=" +
                    String.format(Locale.US, "%.4f", r.igpScore) + " | Bloom hits=" +
                    String.format(Locale.US, "%.2f", r.bloomHitPercent * 100) + "%");
        }
        System.out.println("CSV saved to: " + outputCsv.toAbsolutePath());
    }
}
