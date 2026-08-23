# 🚀 Vector Database from Scratch: Explain Like I'm 5 (ELI5)

Imagine you have a massive library, but the books aren't arranged alphabetically; they are arranged by meaning: sci-fi is right next to books about space, and cooking is next to healthy eating. The `vector-db-from-scratch` project does exactly this, but for a computer!

## 🧠 How the computer understands "meaning"
The computer doesn't know how to read text the way we do, so it turns any information (words, pictures, sentences) into long lists of numbers. These lists are called **vectors**. 

Your database takes these lists of numbers and measures the "distance" between them to see how similar they are. To do this, it uses special mathematical formulas: for example, regular distance (L2) or the angle between vectors (Cosine). If vectors are close to each other, it means their meaning is similar.

## 🗺️ A smart search map (HNSW algorithm)
Searching for the right information among thousands of records one by one is like looking for one specific book by checking every shelf in the world. It takes way too long!

That's why the project uses a cool algorithm called **HNSW**. Imagine you're looking for your house on an online map:
1. First, you look at the whole world from a bird's-eye view (the top layer of the graph) to quickly find the right country.
2. Then you zoom in and go down to lower layers to find the city, then the street.
3. And finally, on the very bottom layer, you find the right house.

This method allows the database to find the closest "neighbors" incredibly fast, skipping over huge distances in milliseconds.

## 🛠️ Why is this pro-level?
Instead of using ready-made building blocks (like popular libraries FAISS or Chroma), the whole system was built completely from scratch. It's like assembling a real car engine with your own hands to thoroughly understand how it works.

* **No cheating:** Only the basic `numpy` library was used for math. No third-party search helpers!
* **Custom save format:** A custom binary file format was invented. The database knows how to save itself to a hard drive and later restore its entire complex graph structure without forgetting anything.
* **Soft-delete:** When you delete data, it doesn't get erased instantly; it gets marked as a "ghost" (tombstone). This is necessary so as not to break the complex search map until it's time to do a "spring cleaning" (`rebuild`).
* **Perfect reliability:** The project is 100% covered by automated tests. Every single line of code was checked for errors.

Essentially, this project shows not just the ability to use ready-made tools, but a deep understanding of complex machine learning algorithms and data structures!
