**Uses of Blockchain in Public Distribution System (PDS) inventory management project**

---

### ✅ Why Use Blockchain in This Project?

#### 1. **Immutable Record-Keeping**

Every time inventory is added, moved, or cleared (FIFO), a blockchain block can be created to record the transaction. These blocks are **immutable**, meaning no one can tamper with the past records. This is very useful in government/public distribution systems where fraud and manipulation are concerns.

#### 2. **Ensures FIFO Compliance**

Each block stores:

- Timestamps
- Batch IDs
- Quantity
- Movement type (stock in, stock out)
- Linked to the previous block (chained)

This makes it easy to **track if items were truly cleared on a First-In-First-Out basis**, since the chain itself shows the order.

#### 3. **Audit Trail**

Anyone (e.g., auditors or government officials) can **trace the full history** of stock movement:

- When was it added to inventory?
- When was it dispatched?
- Was it skipped or misused?

#### 4. **Tamper Detection**

Because each block references the hash of the previous block, **any change in one block breaks the entire chain**. This makes it easy to detect tampering or unauthorized changes.

#### 5. **Builds Trust**

In public systems like ration distribution, blockchain provides a **transparent and verifiable** system that can restore public trust and ensure accountability.

---

### 🧱 Example Use Case:

Let’s say:

- On April 1st, 100kg of rice arrives → Block A is created
- On April 10th, 50kg is dispatched to PDS Shop X → Block B is created referencing Block A
- On April 12th, 20kg more is dispatched from that batch → Block C is created referencing B

These blocks together **prove** that the rice was cleared in FIFO order and no stock was mysteriously lost or used without a record.

---

### ⚙️ In Flask Project:

- When stock is added or removed, a **block is generated** using your blockchain logic and stored in the database.
- You can create an API like `/blockchain/chain` to **view the entire chain**.
- Use hashes and timestamps to ensure data integrity.

---

### Sample Payloads:

/signup - {"email": "rolexman398@gmail.com", "password": "12345", "is_admin": true}

/signin - {"email": "rolexman398@gmail.com", "password": "12345"}

/add_block - {"action": "stock_in", "item_name": "suger", "quantity": "3000kg"}
