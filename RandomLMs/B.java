/**
 * 
 */
// package random;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/*
3
Metals 200 3000
Fertilizers 50 2000
Computers 100 1000
2
Metals 100 2000
Fertilizers 20 1500
 */

public class B {

    private static class Product {
        String name;
        int importCost;
        int importQuantity;
        int manufactureCost;
        int manufactureQuantity;

        public Product(String name, int importCost, int importQuantity, int manufactureCost, int manufactureQuantity) {
            this.name = name;
            this.importCost = importCost;
            this.importQuantity = importQuantity;
            this.manufactureCost = manufactureCost;
            this.manufactureQuantity = manufactureQuantity;
        }

        @Override
        public String toString() {
            return "Product [name=" + name + ", importCost=" + importCost + ", manufactureCost=" + manufactureCost
                    + "]";
        }

    }

    public static void main(String[] args) {
        // TODO Auto-generated method stub
        FastScanner fs = new FastScanner(System.in);

        int N = fs.nextInt();
        List<Product> importedProductsMap = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            String name = fs.next();
            int quantity = fs.nextInt();
            int price = fs.nextInt();
            Product product = new Product(name, price, quantity, 0, 0);
            importedProductsMap.add(product);
        }

        int M = fs.nextInt();
        List<Product> manufactedProductsMap = new ArrayList<>();
        for (int i = 0; i < M; i++) {
            String name = fs.next();
            int quantity = fs.nextInt();
            int price = fs.nextInt();
            Product product = new Product(name, 0, 0, price, quantity);
            manufactedProductsMap.add(product);
        }

        int totalBudgetRequired = getTheIdealBudgetForImporting(importedProductsMap, manufactedProductsMap);

        System.out.println(totalBudgetRequired);
    }

    private static int getTheIdealBudgetForImporting(List<Product> importedProducts,
            List<Product> manufactedProducts) {

        int cost = 0;
        List<Product> tobeRemoveProducts = new ArrayList<>();

        for (Product product : importedProducts) {
            boolean canBeManufacture = false;
            for (Product product2 : manufactedProducts) {
                if (product.name.equalsIgnoreCase(product2.name)) {
                    if (product.manufactureCost < product.importCost) {
                        canBeManufacture = true;
                    }
                    if (canBeManufacture) {
                        System.out.println(product + " --> canbemanufacture --> " + canBeManufacture);
                        cost += calculateTheIdealCostByManufacture(product, product2);
                    } else {
                        cost += calculateTheIdealCostByImporting(product, product2);
                    }

                    tobeRemoveProducts.add(product);
                    tobeRemoveProducts.add(product2);
                    break;
                }
            }
        }

        for (Product product : tobeRemoveProducts) {
            importedProducts.remove(product);
            manufactedProducts.remove(product);
        }

        for (Product product : importedProducts) {
            cost += product.importCost * product.importQuantity;
        }
        for (Product product : manufactedProducts) {
            cost += product.manufactureCost * product.manufactureQuantity;
        }

        return cost;
    }

    private static int calculateTheIdealCostByImporting(Product importProduct, Product manufactureProduct) {
        /*
         * 20 1000
         * 50 2000
         */
        if (importProduct.importQuantity > manufactureProduct.manufactureQuantity) {
            int mc = manufactureProduct.manufactureCost * manufactureProduct.manufactureQuantity;
            int ic = Math.abs(importProduct.importQuantity - manufactureProduct.manufactureQuantity)
                    * importProduct.importCost;
            return mc + ic;
        } else {
            return importProduct.importCost * importProduct.importQuantity;
        }
    }

    private static int calculateTheIdealCostByManufacture(Product importProduct, Product manufactureProduct) {
        /*
         * ** IMPORTING **
         * IM ----- MAN
         * 1. 100 ---- 200
         * 2. 200 ---- 100
         */
        if (importProduct.importQuantity < manufactureProduct.manufactureQuantity) {
            return manufactureProduct.manufactureQuantity * importProduct.importQuantity;
        } else {
            int mc = manufactureProduct.manufactureCost * manufactureProduct.manufactureQuantity;
            int ic = Math.abs(manufactureProduct.manufactureQuantity - importProduct.importQuantity)
                    * importProduct.importCost;
            return mc + ic;
        }
    }

    public static class FastScanner {
        private BufferedReader reader = null;
        private StringTokenizer tokenizer = null;

        public FastScanner(InputStream in) {
            reader = new BufferedReader(new InputStreamReader(in));
            tokenizer = null;
        }

        public String next() {
            if (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    tokenizer = new StringTokenizer(reader.readLine());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            return tokenizer.nextToken();
        }

        public String nextLine() {
            if (tokenizer == null || !tokenizer.hasMoreTokens()) {
                try {
                    return reader.readLine();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }

            return tokenizer.nextToken("\n");
        }

        public long nextLong() {
            return Long.parseLong(next());
        }

        public int nextInt() {
            return Integer.parseInt(next());
        }

        public double nextDouble() {
            return Double.parseDouble(next());
        }

        public int[] nextIntArray(int n) {
            int[] a = new int[n];
            for (int i = 0; i < n; i++)
                a[i] = nextInt();
            return a;
        }

        public long[] nextLongArray(int n) {
            long[] a = new long[n];
            for (int i = 0; i < n; i++)
                a[i] = nextLong();
            return a;
        }
    }
}